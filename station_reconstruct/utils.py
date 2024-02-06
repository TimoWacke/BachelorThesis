import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import netCDF4 as nc
import numpy as np
import os
from tqdm import tqdm
from timezonefinder import TimezoneFinder
import pytz

class DataSet:
    def __init__(self, path, name="", is_utc_time=False):
        self.name = name
        self.path = path

        self.dataset = nc.Dataset(self.path)
    
        self.lon = self.dataset.variables['lon'][:]
        self.lat = self.dataset.variables['lat'][:]
        self.time = self.dataset.variables['time'][:]
        

        unit_str = self.dataset.variables['time'].units
        units = unit_str.split(" ")[0]  # can be hours, days etc.
        try:
            self.start_date = datetime.strptime(unit_str, f"{units} since %Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                self.start_date = datetime.strptime(unit_str, f"{units} since %Y-%m-%dT%H:%M:%S")
            except ValueError:
                self.start_date = datetime.strptime(unit_str, f"{units} since %Y-%m-%d")

        # convert units to seconds
        if units == "hours":
            self.units = 3600
        elif units == "days":
            self.units = 86400
        else:
            raise ValueError(f"Time units {self.units} not yet supported!")
        
        # set first timestep value to 0 and adjust start date accordingly
        self.start_date = self.start_date + timedelta(seconds=int(self.time[0] * self.units))
        self.time = self.time - self.time[0]

        self.is_utc_time = is_utc_time

    def apply_local_time_utc_offset(self, lon, lat):
        if not self.is_utc_time:   
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lng=lon, lat=lat)
            print(f"Timezone for {self.name}: {timezone_str}")
            timezone = pytz.timezone(timezone_str)
            utc_offset = timezone.utcoffset(self.start_date).total_seconds()
            print(f"UTC offset for {self.name}: {utc_offset / 3600} hours")
            utc_startdate = self.start_date - timedelta(seconds=utc_offset)
            self.start_date = utc_startdate
            self.is_utc_time = True

    def intersect(self, other):
        # get the later start date
        start_date = max(self.start_date, other.start_date)
        # get the earlier end date
        end_date = min(self.start_date + timedelta(seconds=int(self.time[-1] * self.units)),
                    other.start_date + timedelta(seconds=int(other.time[-1] * other.units)))
        
        # get the time indices for the intersection



    def get_n_random_times(self, n):
        return sorted(np.random.choice(len(self.time), n, replace=False))

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="right")
        if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
            return idx - 1
        else:
            return idx

    def get_slices_and_area_for_given_gridsize_around_coordinates(self, given_lon, given_lat, width_idx, height_idx):
        given_lon = (given_lon + 360) % 360

        def find_interval(array, value, width):
            if width % 2 == 0:
                min_idx = np.searchsorted(array, value) - (width // 2)
            else:
                min_idx = self.find_nearest(array, value) - width // 2 - 1
            max_idx = min_idx + width
            return min_idx, max_idx

        lon_min_idx, lon_max_idx = find_interval(self.lon, given_lon, width_idx)
        lat_min_idx, lat_max_idx = find_interval(self.lat, given_lat, height_idx)
        area = Area(self.lon[lon_min_idx], self.lon[lon_max_idx], self.lat[lat_min_idx], self.lat[lat_max_idx])

        return slice(lon_min_idx, lon_max_idx), slice(lat_min_idx, lat_max_idx), area

    def get_slices_cropping_a_given_area(self, given_area):
        # Crop longitude
        min_lon_idx = np.searchsorted(self.lon, given_area.min_lon) - 1
        max_lon_idx = np.searchsorted(self.lon, given_area.max_lon)

        # Crop latitude
        min_lat_idx = np.searchsorted(self.lat, given_area.min_lat) - 1
        max_lat_idx = np.searchsorted(self.lat, given_area.max_lat)

        return slice(min_lon_idx, max_lon_idx), slice(min_lat_idx, max_lat_idx)

    def crop_time(self, min_time, max_time):
        min_idx = self.get_time_index(min_time)
        max_idx = self.get_time_index(max_time)
        return slice(min_idx, max_idx)

    def get_time_index(self, given_time):
        # find out the offset time
        offset_time = given_time - self.start_date
        # convert to units
        step = offset_time.total_seconds() / self.units
        # find the nearest time index
        idx =  self.find_nearest(self.time, step)
        return idx

    def time_at_index(self, time_index):
        # calculate human-readable date and time
        date_at_idx = self.start_date + timedelta(seconds=int(self.time[time_index] * self.units))
        return date_at_idx.strftime("%Y-%m-%d %H:%M:%S")
  
    def find_invalid_values(self, var="tas", start_time_idx=None, end_time_idx=None):
        invalid_values = []
        if_slice = slice(start_time_idx, end_time_idx)

        for time_index in tqdm(range(*if_slice.indices(len(self.time))), desc=f"Checking {var}"):
            data = self.dataset.variables[var][time_index, ...].flatten()

            # Use np.isnan directly on the array for NaN check
            nan_check = np.isnan(data)

            # Use a single np.any to check for multiple conditions
            condition_check = np.any([nan_check, data == "", data < 150, data > 350], axis=0)

            if np.any(condition_check):
                invalid_values.append(time_index)

        return invalid_values



class Area:
    def __init__(self, min_lon, max_lon, min_lat, max_lat):
        self.min_lon = (min_lon + 360) % 360
        self.max_lon = (max_lon + 360) % 360
        self.min_lat = min_lat
        self.max_lat = max_lat

    def __str__(self):
        #  lat repr
        def lat_repr(_lat):
            _lat = np.round(_lat, 2)
            if _lat > 0:
                return f"N: {_lat}°"
            else:
                return f"S: {_lat}°"

        # lon repr
        def lon_repr(_lon):
            _lon = np.round(_lon, 2)
            if _lon > 0:
                return f"E: {_lon}°"
            else:
                return f"W: {-_lon}°"

        return f"{lat_repr(self.min_lat)} to {lat_repr(self.max_lat)}, {lon_repr(self.max_lon)} to {lon_repr(self.min_lon)}"

    def __repr__(self):
        return str(self)


class Plot:
    def __init__(self):
        self.dataset = None
        self.area = ""
        self.lon_slice = slice(None)
        self.lat_slice = slice(None)
        self.time_index_list = [0]
        self.vmin = None
        self.vmax = None
        self.get_time_index_list = lambda: self.time_index_list

    def generate_time_index_list(self, n):
        self.time_index_list = self.dataset.get_n_random_times(n)

    def plot(self, vars=["tas"]):
        
        # if not list make it a list
        if not isinstance(vars, list):
            vars = [vars]
        
        for time_index in self.get_time_index_list():
            # set title
            title = self.dataset.name if self.dataset.name else self.dataset.path.split("/")[-1]
            if self.area:
                title += f"\n{self.area}"
            title += f"\n{self.dataset.time_at_index(time_index)}"

            for var in vars:

                # plot
                fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
                # Plot the temperature data with a quadratic colormap
                _lon = self.dataset.lon[self.lon_slice]
                _lat = self.dataset.lat[self.lat_slice]
                _data = self.dataset.dataset.variables[var][time_index, self.lat_slice, self.lon_slice]
                pcm = ax.pcolormesh(_lon, _lat, _data, cmap='viridis', shading='auto', vmin=self.vmin, vmax=self.vmax)

                # Add coastlines
                ax.coastlines()

                # Add colorbar
                cbar = plt.colorbar(pcm, ax=ax, label='Temperature')

                # Set labels and title
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.title(title + (f"\n[{var}]" if len(vars) > 1 else ""))

                # position title a higher
                plt.subplots_adjust(top=1.2)
                
                # Show the plot
                plt.show()


class DatasetPlotter(Plot):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def plot_area(self, area: Area):
        self.area = area
        self.lon_slice, self.lat_slice = self.dataset.crop_area(area)
        self.plot()
        self.lon_slice, self.lat_slice = slice(None), slice(None)

    def plot_grid(self, given_lon, given_lat, width_idx, height_idx):
        self.lon_slice, self.lat_slice, self.area = self.dataset.crop_grid(given_lon, given_lat, width_idx, height_idx)
        self.plot()
        self.lon_slice, self.lat_slice = slice(None), slice(None)


class AreaPlotter(Plot):
    def __init__(self, area: Area):
        super().__init__()
        self.get_time_index_list = None
        self.dataset = None
        self.area = area

    def generate_time_index_list(self, n):
        # overwrite self.get_time_index_list() to use get_n_random_times once self.dataset is set,
        self.get_time_index_list = lambda: self.dataset.get_n_random_times(n)

    def plot_dataset(self, dataset: DataSet):
        self.dataset = dataset
        self.lon_slice, self.lat_slice = self.dataset.crop_area(self.area)
        self.plot()
        self.lon_slice, self.lat_slice = slice(None), slice(None)


class Station:

    def __init__(self, path, name):
        self.name = name
        self.path = path
        self.dataset = DataSet(path, name)


    def get_lon_lat(self):
        return self.dataset.lon[0], self.dataset.lat[0]




if __name__ == "__main__":

    # Set the working directory to the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)

    soltau = Station("./data-sets/soltau.nc", "Soltau")
