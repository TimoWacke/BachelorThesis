import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import netCDF4 as nc
import numpy as np
import os
import shutil

class DataSet:
    def __init__(self, path):
        self.units = None
        self.start_date = None
        self.path = path
        self.dataset = nc.Dataset(path)
        self.lon = self.dataset.variables['lon'][:]
        self.lat = self.dataset.variables['lat'][:]
        self.time = self.dataset.variables['time'][:]

        unit_str = self.dataset.variables['time'].units
        units = unit_str.split(" ")[0]  # can be hours, days etc.
        self.start_date = datetime.strptime(unit_str, f"{units} since %Y-%m-%d %H:%M:%S")

        # convert units to seconds
        if units == "hours":
            self.units = 3600
        elif units == "days":
            self.units = 86400
        else:
            raise ValueError(f"Time units {self.units} not yet supported!")

    def get_n_random_times(self, n):
        return sorted(np.random.choice(len(self.time), n))

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="right")
        if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
            return idx - 1
        else:
            return idx

    def crop_grid(self, given_lon, given_lat, width_idx, height_idx):
        given_lon = (given_lon + 360) % 360
        def find_interval(array, value, width):
            if width % 2 == 0:
                min_idx = np.searchsorted(array, value, side="left") - width // 2 + 1
            else:
                min_idx = self.find_nearest(array, value) - width // 2
            max_idx = min_idx + width
            return min_idx, max_idx
        print("lon interval", find_interval(self.lon, given_lon, width_idx))
        lon_min_idx, lon_max_idx = find_interval(self.lon, given_lon, width_idx)
        lat_min_idx, lat_max_idx = find_interval(self.lat, given_lat, height_idx)
        area = Area(self.lon[lon_min_idx], self.lon[lon_max_idx], self.lat[lat_min_idx], self.lat[lat_max_idx])

        return slice(lon_min_idx, lon_max_idx), slice(lat_min_idx, lat_max_idx), area

    def crop_area(self, given_area):
        # Crop longitude
        min_lon_idx = np.searchsorted(self.lon, given_area.min_lon, side="left")
        max_lon_idx = np.searchsorted(self.lon, given_area.max_lon, side="right")

        # Crop latitude
        min_lat_idx = np.searchsorted(self.lat, given_area.min_lat, side="left")
        max_lat_idx = np.searchsorted(self.lat, given_area.max_lat, side="right")

        return slice(min_lon_idx, max_lon_idx), slice(min_lat_idx, max_lat_idx)


    def crop_time(self, min_time, max_time):
        min_idx = self.get_time_index(min_time)
        max_idx = self.get_time_index(max_time)
        return slice(min_idx, max_idx)

    def get_time_index(self, given_time):
        # find out the offset time
        offset_time = given_time - self.start_date
        # convert to seconds
        return round(offset_time.total_seconds() / self.units)

    def human_readable_time(self, time_index):
        # calculate human-readable date and time
        date_at_idx = self.start_date + timedelta(seconds=self.time[time_index] * self.units)
        return date_at_idx.strftime("%Y-%m-%d %H:%M:%S")

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
        self.area = "full area"
        self.lon_slice = slice(None)
        self.lat_slice = slice(None)
        self.time_index_list = [0]

    def generate_time_index_list(self, n):
        self.time_index_list = self.dataset.get_n_random_times(n)

    def plot(self, var="tas"):
        for time_index in self.time_index_list:
            # set title
            title = self.dataset.path
            title += f"\n{self.area}"
            title += f"\n{self.dataset.human_readable_time(time_index)}"

            # plot
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            # Plot the temperature data with a quadratic colormap
            _lon = self.dataset.lon[self.lon_slice]
            _lat = self.dataset.lat[self.lat_slice]
            _data = self.dataset.dataset.variables[var][time_index, self.lat_slice, self.lon_slice]
            pcm = ax.pcolormesh(_lon, _lat, _data, cmap='viridis', shading='auto')

            # Add coastlines
            ax.coastlines()

            # Add colorbar
            cbar = plt.colorbar(pcm, ax=ax, label='Temperature')

            # Set labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.title(title)

            # Show the plot
            plt.show()


class DatasetPlotter(Plot):
    def __init__(self, path):
        super().__init__()
        self.dataset = DataSet(path)
        self.area = "full area"

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
        self.dataset = None
        self.area = area

    def plot_dataset(self, dataset: DataSet):
            self.dataset = DataSet(dataset.path)
            self.lon_slice, self.lat_slice = self.dataset.crop_area(self.area)


filePlotter = DatasetPlotter("joined.nc")
filePlotter.time_index_list = [9]
hh_lat = 53.55
hh_lon = 9
filePlotter.plot()
# filePlotter.plot_grid(hh_lon, hh_lat, 9, 9)
# filePlotter.plot_area(Area(hh_lon - 4, hh_lon + 4, hh_lat - 8, hh_lat + 4))