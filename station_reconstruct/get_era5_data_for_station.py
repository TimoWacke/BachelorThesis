
from typing import Any
from utils import Station, DataSet

import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor

import xarray as xr
import numpy as np
from tqdm import tqdm
import datetime


class CreateMatchingEra5FileForStationData:

    def __init__(self,
                 station_name: str,
                 station_path: str = "",
                 era5_folder: str = "/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/1hr/atmos/tas/r1i1p1",
                 target_folder: str = "/work/bm1159/XCES/xces-work/k203179/data_sets",
                 grid_width: int = 8,
                 grid_height: int = 8
                 ) -> None:

        self.station_name = station_name
        if station_path == "":
            station_path = "/home/k/k203179/reconstructing-ai-for-weather-station-data/station_reconstruct/station_data_as_nc" + \
                "/" + self.station_name.lower() + ".nc"
        unique_id = "_" + self.station_name.lower()
        self.target_temp_era5_folder = os.path.join(target_folder + "_era5-crop" + unique_id)
        self.target_temp_era5_path = os.path.join(
            target_folder, "era5_for_" + self.station_name.lower() + ".nc")
        self.target_era5_path = os.path.join(
            target_folder, "era5_for_" + self.station_name.lower() + ".nc")
        self.target_station_path = os.path.join(
            target_folder, self.station_name.lower() + ".nc")
        self.station = Station(station_path, station_name)
        # self.station.dataset.apply_local_time_utc_offset(
        #    *self.station.get_lon_lat())
        self.era5_folder = era5_folder
        self.selected_era5_files = []
        self.grid_width = grid_width
        self.grid_height = grid_height

    def extract_areas_from_era5_datasets(self) -> DataSet:

        print(f"choosing example era5 file from {self.era5_folder}")
        # sort by namem, take latest file
        example_era5_file = sorted(os.listdir(self.era5_folder))[-1]
        print("example_era5_file", example_era5_file)
        example_era5_dataset = DataSet(
            os.path.join(self.era5_folder, example_era5_file))

        print("getting slices around:", self.station.get_lon_lat())
        lon_slice, lat_slice, area = example_era5_dataset.get_slices_and_area_for_given_gridsize_around_coordinates(
            *self.station.get_lon_lat(), self.grid_width, self.grid_height
        )

        era5_min_lon_idx, era5_max_lon_idx, _ = lon_slice.indices(
            len(example_era5_dataset.lon))
        era5_min_lat_idx, era5_max_lat_idx, _ = lat_slice.indices(
            len(example_era5_dataset.lat))
        
        print("era5 min / max lin:", era5_min_lon_idx, era5_max_lon_idx)
        print("era5 min / max lat:", era5_min_lat_idx, era5_max_lat_idx)
        
        found_files = os.listdir(self.era5_folder)
        # sort files by name, start with the newes (highes number)
        found_files.sort(reverse=True)

        # yyyymmdd as int for min and max date in self.station.dataset
        min_date = int(self.station.dataset.time_at_index(
            0).split(" ")[0].replace("-", ""))
        max_date = int(
            self.station.dataset.time_at_index(-1).split(" ")[0].replace("-", ""))

        for file in found_files:
            if file.endswith(".nc"):
                era_min_date = int(file.split("_")[-1].split("-")[0]) + 1 # margin so we could use leading and trailing timesteps
                era_max_date = int(file.split("_")[-1].split("-")[1].split(".")[0]) - 1
                if min_date <= era_max_date and max_date >= era_min_date:
                    self.selected_era5_files.append(file)

        # if folder does exist clear it
        if os.path.exists(self.target_temp_era5_folder):
            os.system(f"rm -rf {self.target_temp_era5_folder}")
            
        os.mkdir(self.target_temp_era5_folder)
        # chmod + x
        os.system(f"chmod +x {self.target_temp_era5_folder}")

        # Use tqdm to display a loading bar
        with tqdm(total=len(self.selected_era5_files), desc="Processing Files", unit="file") as pbar:
            def update_progress(*args):
                pbar.update()

            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = []
                for file in self.selected_era5_files:
                    future = executor.submit(
                        crop_and_copy_file,
                        input_file_path=os.path.join(self.era5_folder, file),
                        output_file_path=os.path.join(
                            self.target_temp_era5_folder, file),
                        min_lon=era5_min_lon_idx,
                        max_lon=era5_max_lon_idx,
                        min_lat=era5_min_lat_idx,
                        max_lat=era5_max_lat_idx,
                    )
                    future.add_done_callback(update_progress)
                    futures.append(future)

                # Wait for all tasks to complete
                problematic_files = [
                    future.result() for future in futures if future.result() is not None]

        # Print the filenames of problematic files
        if problematic_files:
            print("Problematic files with grid differences:")
            for problematic_file in problematic_files:
                print(problematic_file)
        else:
            pass
            
        return problematic_files

    def merge_era5_files(self) -> str:

        if os.path.exists(self.target_temp_era5_path):
            os.remove(self.target_temp_era5_path)

        # cdo merge command
        cdo_command = f"cdo cat {self.target_temp_era5_folder}/*.nc {self.target_temp_era5_path}"
        subprocess.run(cdo_command, shell=True)

        return self.target_temp_era5_path

    def transform_era5_to_match_station_time_dimension(self) -> None:
        era5_dataset = DataSet(self.target_temp_era5_path)
        station_dataset = self.station.dataset
        # get later start date
        start_date = max(self.station.dataset.start_date,
                         era5_dataset.start_date)
        # get earlier end date
        end_date = min(
            station_dataset.start_date +
            datetime.timedelta(seconds=int(station_dataset.time[-1] * station_dataset.units)),
            era5_dataset.start_date +
            datetime.timedelta(seconds=int(era5_dataset.time[-1] * era5_dataset.units)),
        )
        
        print(f"station data goes from {station_dataset.start_date} to {station_dataset.start_date + datetime.timedelta(seconds=int(station_dataset.time[-1] * station_dataset.units))}")
        print(f"era5 data goes from {era5_dataset.start_date} to {era5_dataset.start_date + datetime.timedelta(seconds=int(era5_dataset.time[-1] * era5_dataset.units))}")
        
        print("cropping time axis to intersection of station and era5 dataset")
        print("start/enddate (utc):", start_date, "/", end_date)
        
        # get the time indices for the intersection
        station_start_idx = station_dataset.get_time_index(start_date)
        station_end_idx = station_dataset.get_time_index(end_date)

        era5_start_idx = era5_dataset.get_time_index(start_date)
        era5_end_idx = era5_dataset.get_time_index(end_date)
        

        # get invalid values in station dataset
        invalid_station_values = station_dataset.find_invalid_values(
            start_time_idx=station_start_idx,
            end_time_idx=station_end_idx,
        )

        # get invalid values in era5 dataset
        invalid_era5_values = era5_dataset.find_invalid_values(
            start_time_idx=era5_start_idx,
            end_time_idx=era5_end_idx,
        )
        
        print(f"Found {len(invalid_station_values)} invalid values in station dataset")
        print(f"Found {len(invalid_era5_values)} invalid values in era5 dataset")

        # station delete idinces
        station_delete_indices = np.concatenate(
            (invalid_station_values - station_start_idx, invalid_era5_values - era5_start_idx))
        # era5 delete indices
        era5_delete_indices = np.concatenate(
            (invalid_era5_values - era5_start_idx, invalid_station_values - station_start_idx))

        # unique
        station_delete_indices = np.unique(station_delete_indices)
        era5_delete_indices = np.unique(era5_delete_indices)


        station_quick_temp_path = self.station.path + "_temp"
        era5_quick_temp_path = self.target_temp_era5_path + "_temp"        
        

        if os.path.exists(station_quick_temp_path):
            os.remove(station_quick_temp_path)
    
    
        # delete leading and trailing times outside of intersection
        cdo_command = f"cdo seltimestep,{station_start_idx + 1}/{station_end_idx + 1} {self.station.path} {station_quick_temp_path}"
        subprocess.run(cdo_command, shell=True)


        if os.path.exists(era5_quick_temp_path):
            os.remove(era5_quick_temp_path)
        cdo_command = f"cdo seltimestep,{era5_start_idx + 1}/{era5_end_idx + 1} {self.target_temp_era5_path} {era5_quick_temp_path}"
        subprocess.run(cdo_command, shell=True)


        def cdo_delete_in_batches(original_path, delete_indices, batch_size=1000):
            n = len(delete_indices) / batch_size
            n = int(max(1, n))
            delete_indices_batches = np.array_split(delete_indices, n, axis=0)
            count_deleted = 0
            with tqdm(total=len(delete_indices)) as pbar:
                for delete_indices_batch in delete_indices_batches:
                    delete_indices_batch = delete_indices_batch - count_deleted
                    cdo_command = f"cdo delete,timestep={','.join(map(str, delete_indices_batch + 1))} {original_path} {original_path}_t"
                    subprocess.run(cdo_command, shell=True)
                    # move
                    subprocess.run(f"mv {original_path}_t {original_path}", shell=True)
                    count_deleted += len(delete_indices_batch)
                    pbar.update(len(delete_indices_batch))

            print(f"Deleted {count_deleted} timesteps from {original_path} dataset because of invalid values")

        cdo_delete_in_batches(station_quick_temp_path, station_delete_indices)
        cdo_delete_in_batches(era5_quick_temp_path, era5_delete_indices)
            
        if os.path.exists(self.target_station_path):
                os.remove(self.target_station_path)
        if os.path.exists(self.target_era5_path):
                os.remove(self.target_era5_path)
                
        # move files to target
        shutil.move(station_quick_temp_path, self.target_station_path)
        shutil.move(era5_quick_temp_path, self.target_era5_path)
        return self.target_station_path, self.target_era5_path


def crop_and_copy_file(input_file_path, output_file_path, min_lon, max_lon, min_lat, max_lat):
   
    if not os.path.exists(output_file_path):
        # cdo selindexbox with subprocess
        cdo_command = f"cdo selindexbox,{min_lon + 1},{max_lon},{min_lat + 1},{max_lat} {input_file_path} {output_file_path}"
        subprocess.run(cdo_command, shell=True)

    check_data = xr.open_dataset(output_file_path)

    check_data.close()
