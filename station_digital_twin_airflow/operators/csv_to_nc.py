from airflow.models import BaseOperator

class CSVtoNC(BaseOperator):
    
    def __init__(self,
                 task_id,
                 station_name,
                 *args,
                 **kwargs):
        self.task_id = task_id
        self.station_name = station_name
        
    
        super().__init__(task_id=task_id, *args, **kwargs)

    def extract(self):
        ...
        
    def transform(self):
        ...
    
    def load(self):
        ...

    def execute(self, context):
        ...
        
"""
------------------------------------------------------------
"""

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

class DatToNcConverter:

    def __init__(self, name, directory = None, target_directory = None):
        self.name = name
        self.directory = directory if directory is not None else os.getcwd() + "/station_data_as_dat/" + self.name.capitalize()
        self.target_directory = target_directory if target_directory is not None else os.getcwd() + "/station_data_as_nc/"
        self.files = self.get_files()
        self.dataframe = None
        self.nc_data = None
        self.meta_data = self.extract_meta_data()

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


        def custom_aggregation(series):
            # If all values are the same or NaN, return NaN; otherwise, return the mean
            if series.nunique() <= 2:
                return np.nan
            else:
                return np.median(series)

        # merge all minutely data into one row using the mean
        hourly_df = df.resample("H").apply(custom_aggregation)

        # fill the non-NaN value into the temp column
        hourly_df["temp"] = hourly_df[["mcp9808"]].mean(axis = 1)

        return hourly_df
        

    def transform(self):
        
        # interesting columns in dataframe
        mapping = {
            "temp": "tas",
            "vis_light": "vis_light",
            "uv_light": "uv_light",
            "ir_light": "ir_light",
        }

        # drop columns not in mapping
        self.dataframe = self.dataframe[list(mapping.keys())]

        # rename columns
        self.dataframe = self.dataframe.rename(columns = mapping)
        
        # convert temp from C to K
        self.dataframe["tas"] = self.dataframe["tas"] + 273.15
        
    
    def load(self, location):

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