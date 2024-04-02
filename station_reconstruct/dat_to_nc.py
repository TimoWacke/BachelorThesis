"""
Converts all .dat files in a directory to a .nc file in the working directory.

Modules used:

    * os
    * numpy
    * xarray
    * datetime
    * pandas
    * tqdm (for loading bar)
    * re (for meta data extraction)

"""

import os
from typing import Any
from typing_extensions import SupportsIndex
import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd
import tqdm
import re

from map_minutes_to_grid import mapping_rule

class DatToNcConverter:

    def __init__(self, name, directory = None, target_directory = None, hourly = False,
                 grid_blueprint = None):
        self.name = name
        self.directory = directory if directory is not None else os.getcwd() + "/station_data_as_dat/" + self.name.capitalize()
        self.target_directory = target_directory if target_directory is not None else os.getcwd() + "/station_data_as_nc/"
        self.files = self.get_files()
        self.dataframe = None
        self.nc_data = None
        self.meta_data = self.extract_meta_data()
        self.hourly = hourly
        self.grid_blueprint = grid_blueprint

    # determine files in directory

    def get_files(self):

        files = []
        for file in os.listdir(self.directory):
            if file.endswith(".dat"):
                files.append(file)

        # sort files by name
        return sorted(files)
    
    # convert .dat file to dataframe and append to dataframe

    def convert_to_dataframe(self, file) -> pd.DataFrame:
        
        # load into pandas dataframe, first line are the column names
        df = pd.read_csv(self.directory + "/" + file, sep = "\s+", header = 0)
    
        return self.transform_partial(df)

    # append dataframe to the end of self.dataframe
    def append_to_dataframe(self, df):
        self.dataframe = pd.concat([self.dataframe, df])


    def extract_meta_data(self):
        meta_data = {}

        # Define patterns for extracting relevant information
        location_pattern = re.compile(r'Location: ([\d.-]+) deg Lat, ([\d.-]+) deg Lon')
        elevation_pattern = re.compile(r'Elevation: (\d+) m')

        # Search for .rtf files in the directory
        rtf_files = [file for file in os.listdir(self.directory) if file.endswith('.rtf')]

        if not rtf_files:
            print("Error: No .rtf files found in the directory.")
            return meta_data

        # Take the first .rtf file found
        rtf_file_path = os.path.join(self.directory, rtf_files[0])

        try:
            with open(rtf_file_path, 'r') as file:
                content = file.read()

                # Extract coordinates
                match_location = location_pattern.search(content)
                if match_location:
                    latitude = float(match_location.group(1))
                    longitude = float(match_location.group(2))
                    meta_data['latitude'] = latitude
                    meta_data['longitude'] = longitude

                # Extract elevation
                match_elevation = elevation_pattern.search(content)
                if match_elevation:
                    elevation = int(match_elevation.group(1))
                    meta_data['elevation'] = elevation

        except FileNotFoundError:
            print(f"Error: File {rtf_file_path} not found.")

        return meta_data


    # extract a whole folder of .dat files into to self.dataframe

    def extract(self, first_n_files = None):
        # initialize dataframe
        self.dataframe = pd.DataFrame()

        # loading bar for progress
        if first_n_files is None:
            first_n_files = len(self.files)
        for file in tqdm.tqdm(self.files[:first_n_files]):
            df = self.convert_to_dataframe(file)
            self.append_to_dataframe(df)
        return self.dataframe
    
    # convert dataframe to netcdf compatible format datatype

    def transform_partial(self, df):
         # convert year mon day hour min columns to datetime object (as int)
        df["datetime"] = df.apply(lambda row: datetime(int(row["year"]), int(row["mon"]), int(row["day"]), int(row["hour"]), int(row["min"])), axis = 1)
        # drop year mon day hour min columns
        df = df.drop(columns = ["year", "mon", "day", "hour", "min"])

        # convert all -999.99 values to NaN
        df = df.replace(-999.99, np.nan)

        # set datetime column as index
        df = df.set_index("datetime")


        # use certain sensors
        df["temp"] = df[["mcp9808"]].mean(axis = 1)
                
        # convert temp from C to K
        df["temp"] = df["temp"] + 273.15
        

        def custom_aggregation(series):
            # If all values are the same or NaN, return NaN; otherwise, return the mean
            if series.nunique() <= 2:
                return np.nan
            else:
                return np.median(series)

        if self.hourly:
            # merge all minutely data into one row using the mean
            hourly_df = df.resample("H").apply(custom_aggregation)
        else:
            
            hourly_df = pd.DataFrame(columns = ["temp"])
            
            for hour, hour_data in df.resample("H"):
                hourly_temp_array = np.nan * np.zeros((8, 8))
                
                for minute, temp in zip(hour_data.index.minute, hour_data["temp"].values):
                    row, col = mapping_rule[minute]
                    hourly_temp_array[row, col] = temp
                    
                hourly_df.loc[hour, 'temp'] = hourly_temp_array

        return hourly_df
        

    def transform(self):
        
        # interesting columns in dataframe
        mapping = {
            "temp": "tas",
            "vis_light": "vis_light",
            "uv_light": "uv_light",
            "ir_light": "ir_light",
        }
        
        # intersection of columns in dataframe and mapping
        intersect_columns = list(set(self.dataframe.columns).intersection(set(mapping.keys())))

        # drop columns not in mapping
        self.dataframe = self.dataframe[intersect_columns]
        
        if self.hourly:
            self.dataframe = self.dataframe.dropna(subset=["temp"])

        # rename columns
        self.dataframe = self.dataframe.rename(columns = mapping)

    
    def load(self, location):
        if self.hourly:
            print(self.dataframe["tas"].values.shape)
            ds = xr.Dataset(
                {
                    "tas": (["time", "lat", "lon"], self.dataframe["tas"].values.reshape(-1, 1, 1)),
                },
                coords={
                    "time": self.dataframe.index.values,
                    "lat": [self.meta_data["latitude"]],
                    "lon": [self.meta_data["longitude"]],
                },
            )
        else:
            # tas column is an 8x8 array
            # write 8x8 grid in the netcdf file
            blueprint_ds = xr.open_dataset(self.grid_blueprint)
            lats = blueprint_ds.lat.values
            lons = blueprint_ds.lon.values
            print(self.dataframe["tas"].values.shape)
            ds = xr.Dataset(
                {
                    "tas": (["time", "lat", "lon"], [grid for grid in self.dataframe["tas"].values]),
                },
                coords={
                    "time": self.dataframe.index.values,
                    "lat": lats,
                    "lon": lons,
                },
            )
            
            

        save_to_path = location + self.name.lower() + ".nc"
        print(f"Saving to {save_to_path}")
        
        # if file already exists, delete it
        if os.path.exists(save_to_path):
            os.remove(save_to_path)

        # Save the xarray Dataset to NetCDF
        ds.to_netcdf(save_to_path)
        

    def execute(self, location=None):
        self.extract()
        self.transform()
        if location is None:
            location = self.target_directory
        self.load(location)