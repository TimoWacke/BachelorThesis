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
        
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        for folder in [self.test_folder, self.train_folder, self.val_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        for folder in [self.test_folder, self.train_folder, self.val_folder]:
            subprocess.run(f"rm {folder}/*{self.station_file_name[:-3]}*", shell=True)

    def assert_file_compability(self):
        self_check = DataSet(self.station_file, "self_check")
        other_check = DataSet(self.reanalysis_file, "other_check")

        # assert and if fails print differences
        assert np.array_equal(self_check.time, other_check.time), f"Time arrays are not equal! {len(self_check.time)} != {len(other_check.time)}"

    def prepare_trainings_files(self):

        # crop test files to one year
        min_idx, max_idx = self.find_year_indices(self.year)
        
        # copy from data_sets, to test
        subprocess.run(f"cp {self.reanalysis_file} {self.test_folder}/{self.reanalysis_file_name}", shell=True)
        subprocess.run(f"cp {self.station_file} {self.test_folder}/{self.station_file_name}", shell=True)

        # copy test year to val
        cdo_command = f"cdo seltimestep,{min_idx+1}/{max_idx+1} {self.reanalysis_file} {self.val_folder}/{self.reanalysis_file_name}"
        subprocess.run(cdo_command, shell=True)
        # grid format as template for expected station file
        subprocess.run(f"cp {self.val_folder}/{self.reanalysis_file_name} {self.val_folder}/expected_{self.station_file_name}", shell=True)
        cdo_command = f"cdo seltimestep,{min_idx+1}/{max_idx+1} {self.station_file} {self.val_folder}/{self.station_file_name}"
        subprocess.run(cdo_command, shell=True)
        
        # and everything except the year to train
        cdo_command = f"cdo delete,timestep={min_idx+1}/{max_idx+1} {self.reanalysis_file} {self.train_folder}/{self.reanalysis_file_name}"
        subprocess.run(cdo_command, shell=True)
        # grid format as template for expected station file
        subprocess.run(f"cp {self.train_folder}/{self.reanalysis_file_name} {self.train_folder}/expected_{self.station_file_name}", shell=True)
        cdo_command = f"cdo delete,timestep={min_idx+1}/{max_idx+1} {self.station_file} {self.train_folder}/{self.station_file_name}"
        subprocess.run(cdo_command, shell=True)
        
        grid_files_to_be_filled_with_station = [f"{self.val_folder}/expected_{self.station_file_name}", f"{self.train_folder}/expected_{self.station_file_name}"]
        station_reference_files = [f"{self.val_folder}/{self.station_file_name}", f"{self.train_folder}/{self.station_file_name}"]

        for grid, station in zip(grid_files_to_be_filled_with_station, station_reference_files):
            with nc.Dataset(station, "r") as measurements:
                with nc.Dataset(grid, "r+") as dataset:
                    try:
                        # use progress bar
                        for time_index in tqdm(range(len(dataset.variables["time"])), desc=f"preparing {station}"):
                            dataset.variables["tas"][time_index, ...] = measurements.variables["tas"][time_index, ...]
                    except KeyError:
                        print(grid, station)
                        print(dataset.variables.keys())
                        print(measurements.variables.keys())
            os.remove(station)
    
        # create a station file filled with nans in test folder called cleaned_ ... the expected file will be deleted 
        # copy to get test template for station with the grid dimension from era5
        shutil.copyfile(f"{self.test_folder}/{self.reanalysis_file_name}", f"{self.test_folder}/temp_{self.station_file_name}")        
    
        import xarray as xr
        with xr.open_dataset(f"{self.test_folder}/temp_{self.station_file_name}", decode_times=False) as dataset:
            dataset.tas.values[:] = np.nan
            dataset.to_netcdf(f"{self.test_folder}/cleaned_{self.station_file_name}")
        
        if os.path.exists(f"{self.test_folder}/temp_{self.station_file_name}"):
            os.remove(f"{self.test_folder}/temp_{self.station_file_name}")
        
        for_the_gap_path = self.reanalysis_file.split(self.station_file_name.split(".")[0])[0] + f"the_gaps_{self.station_file_name}"
        print(for_the_gap_path)
        if os.path.exists(for_the_gap_path):
            shutil.copyfile(for_the_gap_path, f"{self.test_folder}/temp_era5_for_the_gaps_{self.station_file_name}")
            with xr.open_dataset(f"{self.test_folder}/temp_era5_for_the_gaps_{self.station_file_name}", decode_times=False) as dataset:
                dataset.tas.values[:] = np.nan
                dataset.to_netcdf(f"{self.test_folder}/cleaned_era5_for_the_gaps_{self.station_file_name}")
        
            os.system(f"mv {self.test_folder}/temp_era5_for_the_gaps_{self.station_file_name} {self.test_folder}/era5_for_the_gaps_{self.station_file_name}")
            
        self.split_trainings_files_by_variables(["year", "intra_year", "intra_day", "leading_tas", "trailing_tas"])
     
    def split_trainings_files_by_variables(self, variables):
        for folder in [self.test_folder, self.train_folder, self.val_folder]:
            available_variables = DataSet(f"{folder}/{self.reanalysis_file_name}").dataset.variables.keys()
            for var in variables:
                if var not in available_variables:
                    continue
                print(f"Splitting {var} in {folder}")
                cdo = f"cdo selvar,{var} {folder}/{self.reanalysis_file_name} {folder}/{var}_at_{self.station_file_name}"
                subprocess.run(cdo, shell=True)  

    def find_year_indices(self, year):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        station_dataset = DataSet(self.station_file, "station")
        crop_slice = station_dataset.crop_time(start_date, end_date)
        min_idx = crop_slice.start
        max_idx = crop_slice.stop

        return min_idx, max_idx - 1