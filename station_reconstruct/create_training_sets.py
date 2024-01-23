import os
import numpy as np

import netCDF4 as nc
import shutil
import subprocess

from datetime import datetime
from tqdm import tqdm

from utils import DataSet

class TrainingsFilePair:

    def __init__(self,
                 station_file,
                 reanalysis_file,
                 data_folder="/work/bm1159/XCES/xces-work/k203179/data",
                 test_year=2021,
                 val_count=16,
                 ):
        self.station_file = station_file
        self.reanalysis_file = reanalysis_file
        self.station_file_name = os.path.basename(station_file)
        self.reanalysis_file_name = os.path.basename(reanalysis_file)
        self.data_folder = data_folder
        self.year = test_year
        self.val = val_count
        self.assert_file_compability()
        self.test_folder = f"{self.data_folder}/test"
        self.train_folder = f"{self.data_folder}/train"
        self.val_folder = f"{self.data_folder}/val"

        if os.path.exists(self.data_folder):
            # remove old files
            os.system(f"rm -rf {self.data_folder}")

        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
            os.mkdir(f"{self.data_folder}/test")
            os.mkdir(f"{self.data_folder}/train")
            os.mkdir(f"{self.data_folder}/val")

    def assert_file_compability(self):
        self_check = DataSet(self.station_file, "self_check")
        other_check = DataSet(self.reanalysis_file, "other_check")

        # assert and if fails print differences
        assert np.array_equal(self_check.time, other_check.time), f"Time arrays are not equal! {len(self_check.time)} != {len(other_check.time)}"

    def prepare_trainings_files(self):

        # crop test files to one year
        min_idx, max_idx = self.find_year_indices(self.year)
        # cdo command
        cdo_command = f"cdo seltimestep,{min_idx}/{max_idx} {self.reanalysis_file} {self.test_folder}/{self.reanalysis_file_name}"
        subprocess.run(cdo_command, shell=True)
        cdo_command = f"cdo seltimestep,{min_idx}/{max_idx} {self.station_file} {self.test_folder}/reality_{self.station_file_name}"
        subprocess.run(cdo_command, shell=True)
        # creating template for station with the grid dimension from era5 - in this test case will be filled with nans
        shutil.copyfile(f"{self.test_folder}/{self.reanalysis_file_name}", f"{self.test_folder}/expected_{self.station_file_name}")

        # copy temp without the year
        shutil.copyfile(self.reanalysis_file, f"{self.val_folder}/temp_{self.reanalysis_file_name}")
        shutil.copyfile(self.station_file, f"{self.val_folder}/temp_{self.station_file_name}")

        # copy random val values
        temp_dataset = DataSet(f"{self.val_folder}/temp_{self.station_file_name}", "temp")
        val_time_indices = temp_dataset.get_n_random_times(self.val)

        # copy cdo select times
        cdo = f"cdo seltimestep,{','.join(map(str, val_time_indices))} {self.val_folder}/temp_{self.reanalysis_file_name} {self.val_folder}/{self.reanalysis_file_name}"
        subprocess.run(cdo, shell=True)
        cdo = f"cdo seltimestep,{','.join(map(str, val_time_indices))} {self.val_folder}/temp_{self.station_file_name} {self.val_folder}/{self.station_file_name}"
        subprocess.run(cdo, shell=True)
        # creating template for station with the grid dimension from era5
        shutil.copyfile(f"{self.val_folder}/{self.reanalysis_file_name}", f"{self.val_folder}/expected_{self.station_file_name}")
        
        # copy the rest to train
        cdo = f"cdo delete,timestep={','.join(map(str, val_time_indices))} {self.val_folder}/temp_{self.reanalysis_file_name} {self.train_folder}/{self.reanalysis_file_name}"
        subprocess.run(cdo, shell=True)
        cdo = f"cdo delete,timestep={','.join(map(str, val_time_indices))} {self.val_folder}/temp_{self.station_file_name} {self.train_folder}/{self.station_file_name}"
        subprocess.run(cdo, shell=True)        
        # creating template for station with the grid dimension from era5
        shutil.copyfile(f"{self.train_folder}/{self.reanalysis_file_name}", f"{self.train_folder}/expected_{self.station_file_name}")
        
        os.remove(f"{self.val_folder}/temp_{self.reanalysis_file_name}")

        grid_files_to_be_filled_with_station = [f"{self.val_folder}/expected_{self.station_file_name}", f"{self.train_folder}/expected_{self.station_file_name}"]
        station_reference_files = [f"{self.val_folder}/{self.station_file_name}", f"{self.train_folder}/{self.station_file_name}"]

        for grid, station in zip(grid_files_to_be_filled_with_station, station_reference_files):
            with nc.Dataset(station, "r") as measurements:
                with nc.Dataset(grid, "r+") as dataset:
                    try:
                        # use progress bar
                        for time_index in tqdm(range(len(dataset.variables["time"])), desc=f"preparing {station}"):
                            dataset.variables["tas"][time_index, : :] = measurements.variables["tas"][time_index, 0, 0]
                    except KeyError:
                        print(grid, station)
                        print(dataset.variables.keys())
                        print(measurements.variables.keys())
            os.remove(station)
            
    
        import xarray as xr
        with xr.open_dataset(f"{self.test_folder}/expected_{self.station_file_name}", decode_times=False) as dataset:
            dataset.tas.values[:] = np.nan
            dataset.to_netcdf(f"{self.test_folder}/cleaned_{self.station_file_name}")
        
        if os.path.exists(f"{self.test_folder}/expected_{self.station_file_name}"):
            os.remove(f"{self.test_folder}/expected_{self.station_file_name}")
            

    def find_year_indices(self, year):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        station_dataset = DataSet(self.station_file, "station")
        crop_slice = station_dataset.crop_time(start_date, end_date)
        min_idx = crop_slice.start
        max_idx = crop_slice.stop

        return min_idx, max_idx
        
     


trainingsFilePreparer = TrainingsFilePair("/work/bm1159/XCES/xces-work/k203179/data_sets/marshall.nc", "/work/bm1159/XCES/xces-work/k203179/data_sets/era5_for_marshall.nc")
trainingsFilePreparer.prepare_trainings_files()
