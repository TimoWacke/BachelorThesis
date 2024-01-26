
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import os

from tqdm import tqdm

class TimeAndDateContext:
    
    def __init__(self, width = 8, height = 8, scalar = 1):
        self.width = width
        self.height = height
        self.scalar = scalar
    
    
    def generate_time_context_variables_in_dataset(self, path : str):
        
        """
        apply the overlay on the given dataset and variable
        """
        # open dataset in append mode
        dataset = xr.open_dataset(path)
        
        dataset["year"] = (("time", "lon", "lat"), np.zeros((len(dataset.time), len(dataset.lon), len(dataset.lat))))
        dataset["intra_year"] = (("time", "lon", "lat"), np.zeros((len(dataset.time), len(dataset.lon), len(dataset.lat))))
        dataset["intra_day"] = (("time", "lon", "lat"), np.zeros((len(dataset.time), len(dataset.lon), len(dataset.lat))))
        
        
        
        with tqdm(total=len(dataset.time)) as pbar:
            for i in range(len(dataset.time)):
                
               # set all values at timestep...
                
                # in year to the year of the timestep - 2000
                dataset["year"][i] = dataset.time[i].dt.year - 2000
                
                half_width = int(self.width/2)
                
                # in intra_year upper half sinus and lower half cosinus of the day of the year of the timestep / 365 * 2 pi
                dataset["intra_year"][i, :, :] = np.where(
                    np.arange(self.width)[None, :] < half_width,
                    np.sin(dataset.time[i].dt.dayofyear / 365 * 2 * np.pi),
                    np.cos(dataset.time[i].dt.dayofyear / 365 * 2 * np.pi)
                )
                
                # in intra_day upper half sinus and lower half cosinus of the hour of the timestep / 24 * 2 pi
                dataset["intra_day"][i, :, :] = np.where(
                    np.arange(self.width)[None, :] < half_width,
                    np.sin(dataset.time[i].dt.hour / 24 * 2 * np.pi),
                    np.cos(dataset.time[i].dt.hour / 24 * 2 * np.pi)
                )
                
                # update progress bar
                pbar.update(1)
        
        # save the modified dataset
        dataset.to_netcdf(path + ".tmp")
        os.system("mv " + path + ".tmp " + path)
                
        # display the result as a pandas dataframe
        df = pd.DataFrame({
            "year": dataset["year"][:, 0, 0],
            "intra_year_1": dataset["intra_year"][:, -1, 0],
            "intra_year_2": dataset["intra_year"][:, 0, -1],
            "intra_day_1": dataset["intra_day"][:, -1, 0],
            "intra_day_2": dataset["intra_day"][:, 0, -1]
        })
        df.index = dataset.time
        
        print(df)

if __name__ == "__main__":

    # Example usage:
    test_time_and_date_context = TimeAndDateContext()
    test_path = "/work/bm1159/XCES/xces-work/k203179/data_sets/era5_for_vienna.nc"
    test_time_and_date_context.generate_time_context_variables_in_dataset(test_path)
    dataset_review = xr.open_dataset(test_path + ".tmp")
    print(dataset_review)