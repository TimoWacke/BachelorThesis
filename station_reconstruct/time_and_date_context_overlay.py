
import types
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import os

from utils import DataSet
from tqdm import tqdm

class TimeAndDateContextOverlay:
    
    def __init__(self, width = 8, height = 8, scalar = 0.5):
        self.width = width
        self.height = height
        self.scalar = scalar
    
    def day_overlay(self, data: np.ndarray, timestamp : datetime) -> np.ndarray:
        """
        initial basic pattern inserts vertical lines into every second column
        """
        day_overlay = self.get_day_overlay_strength(timestamp)
        for i in range(len(data[0])):
            if i % 2 == 0:
                data[:, i] += day_overlay
        
        return data
        
        
    def year_overlay(self, data: np.ndarray, timestamp : datetime) -> np.ndarray:
        """
        initial basic pattern inserts horizontal lines into every second row
        """
        year_overlay = self.get_year_overlay_strength(timestamp)
        
        for i in range(len(data)):
            if i % 2 == 0:
                data[i, :] += year_overlay     
        
        return data 
        
        
    def get_day_overlay_strength(self, timestamp : datetime) -> float:
        """
        periodic function to map the time of day to an overlay value
        within self.scalar
        """
        return np.sin(timestamp.hour / 24 * np.pi) * self.scalar
    
    def get_year_overlay_strength(self, timestamp : datetime) -> float:
        """
        periodic function to map the time of year to an overlay value
        within self.scalar
        """
        day_of_year = timestamp.timetuple().tm_yday
        return np.sin(day_of_year / 365 * np.pi) * self.scalar
    
    def apply_on_dataset(self, path : str):
        
        """
        apply the overlay on the given dataset and variable
        """
        # open dataset in read and write mode
        os.system(f"chmod +x {path}")
        nc_dataset = nc.Dataset(path)
        data_set = DataSet(path) 
        
        with tqdm(total=len(data_set.time)) as pbar:
            for idx, time in enumerate(data_set.time):
                timestamp = data_set.start_date + timedelta(seconds=int(time * data_set.units))
                nc_dataset.variables["tas"][idx, :, :] = self.day_overlay(nc_dataset.variables["tas"][idx, :, :], timestamp)
                nc_dataset.variables["tas"][idx, :, :] = self.year_overlay(nc_dataset.variables["tas"][idx, :, :], timestamp)
                pbar.update(1)