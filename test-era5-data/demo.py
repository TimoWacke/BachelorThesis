import os
from climatereconstructionai import evaluate, train
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import shutil
import subprocess

os.environ["ESMFMKFILE"] = "/Users/timo.wacke/anaconda3/envs/crai/lib/esmf.mk"


def join_climate_data(folder_path):
    """
    Joins all climate data in folder_path into a single netCDF file
    Using cdo cat command in subprocess

    @param folder_path: path to folder containing climate data
    """

    # 1. clear joined.nc if it exists
    # 2. run cdo cat *.nc joined.nc inside the folder
    # 3. add permissions to joined.nc

    if os.path.exists(f"{folder_path}/joined.nc"):
        os.remove(f"{folder_path}/joined.nc")

    cdo_command = f"cdo cat *.nc joined.nc"
    subprocess.run(cdo_command, shell=True, cwd=folder_path)
    subprocess.run(f"chmod 666 {folder_path}/joined.nc", shell=True)


def input_to_expected_output(input_path):
    """
    Transforms input file according to the expected output
    @param input_path: path to input file
    @return: None, changes are made to the file
    """

    # 1. copy file to *_expected.nc
    # 2. open file
    # 3. for each timestep apply the transformation

    # current transformation sets to median of all values

    output_path = input_path.replace(".nc", "_expected.nc")
    # copy file
    shutil.copy(input_path, output_path)

    with nc.Dataset(output_path, 'r+') as dataset:
        # for each timestep
        for i in range(len(dataset.variables['time'][:])):
            # set all lat and long values to the value of the 0 lat and 0 long value
            dataset.variables['tas'][i, :, :] = np.median(dataset.variables['tas'][i, :, :])


# binary search to find the nearest latitude and longitude values
def find_nearest(array, value, visualize=False):
    def show(*args):
        if visualize:
            print(*args)

    show("find nearest to value:", value)

    idx = np.searchsorted(array, value, side="right")

    show("found idx to right:", idx)
    # print array [first 3 values, ... x, idx-2:u, idx-1:v, idx:w, idx+1:x, idx+2:y, ... last 3 values]
    # float to 2 decimal places
    show_array = np.array([round(x, 2) for x in array])
    show(show_array[:3], "...", f"{idx-2}: {show_array[idx-2]}", f"{idx-1}: {show_array[idx-1]}", f"{idx}: {show_array[idx]}", f"{idx+1}: {show_array[idx+1]}", f"{idx+2}: {show_array[idx+2]}", "...", show_array[-3:])
    if idx > 0 and np.abs(value - array[idx - 1]) < np.abs(value - array[idx]):
        show(f"returning {idx - 1}")
        return idx - 1
    else:
        show(f"returning {idx}")
        return idx


# unit test find nearest
array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
value = 3.4
print(find_nearest(array, value, visualize=True))


def create_train_datasets_from_era5(file_path, given_lat, given_lon):
    # open the netCDF file
    with nc.Dataset(file_path, 'r+') as dataset:
        # get the variables
        lat_var = dataset.variables['lat'][:]
        lon_var = dataset.variables['lon'][:]
        time_var = dataset.variables['time'][:]

        lat_idx = find_nearest(lat_var[:], given_lat)
        lon_idx = find_nearest(lon_var[:], given_lon)

        # 9x9 grid around the given latitude and longitude values
        lat_idx_min = lat_idx - 4
        lat_idx_max = lat_idx + 3
        lon_idx_min = lon_idx - 4
        lon_idx_max = lon_idx + 3

        # append p % of the time-steps, max 10
        p = 0.1
        mask_n = min(int(len(time_var[:]) * p), 10)
        # get n random different idx values in the range of the time_var
        mask_idx = np.random.choice(len(time_var[:]), mask_n, replace=False)
        # sort them
        mask_idx = np.sort(mask_idx)
        # add 1 to each of the idx values
        mask_idx = mask_idx + 1
        print(mask_idx)

        # cut out the 9x9 grid
        cdo_command = f'cdo selindexbox,{lon_idx_min},{lon_idx_max},{lat_idx_min},{lat_idx_max} {file_path} temp.nc'
        print(cdo_command)
        subprocess.call(cdo_command, shell=True)
        # create training timesteps and evalutation timesteps
        cdo_command = f'cdo delete,timestep={",".join(map(str, mask_idx))} temp.nc ../data/train/9x9_training.nc'
        subprocess.call(cdo_command, shell=True)
        cdo_command = f'cdo seltimestep,{",".join(map(str, mask_idx))} temp.nc ../data/val/9x9_training.nc'
        subprocess.call(cdo_command, shell=True)

        # create expected output
        input_to_expected_output("../data/train/9x9_training.nc")
        input_to_expected_output("../data/val/9x9_training.nc")

        # delete temp.nc
        os.remove("temp.nc")


def create_usage_dataset_from_era5(file_path, given_lat, given_lon, time_steps):
    # open the netCDF file
    with nc.Dataset(file_path, 'r+') as dataset:
        # get the variables
        lat_var = dataset.variables['lat'][:]
        lon_var = dataset.variables['lon'][:]

        lat_idx = find_nearest(lat_var[:], given_lat)
        lon_idx = find_nearest(lon_var[:], given_lon)

        # 9x9 grid around the given latitude and longitude values
        lat_idx_min = lat_idx - 4
        lat_idx_max = lat_idx + 3
        lon_idx_min = lon_idx - 4
        lon_idx_max = lon_idx + 3

        # cut out the 9x9 grid
        cdo_command = f'cdo selindexbox,{lon_idx_min},{lon_idx_max},{lat_idx_min},{lat_idx_max} {file_path} temp.nc'
        subprocess.run(f"chmod 666 joined.nc", shell=True)
        print(cdo_command)
        # subprocess.call(cdo_command, shell=True)
        # create training timesteps and evalutation timesteps
        cdo_command = f'cdo seltimestep,{",".join(map(str, time_steps))} temp.nc ../data/test/9x9_test.nc'
        subprocess.call(cdo_command, shell=True)

        # print expected output (using median)
        with nc.Dataset("../data/test/9x9_test.nc", 'r+') as dataset:
            for idx in range(len(dataset.variables['time'][:])):
                # calculate human-readable date and time
                # get step, and start date
                hours = float(dataset.variables['time'][idx])
                start_date = datetime.datetime.strptime(dataset.variables['time'].units,
                                                        "hours since %Y-%m-%d %H:%M:%S")
                # add hours to start date
                date = start_date + datetime.timedelta(hours=hours)
                median = np.median(dataset.variables['tas'][idx, :, :])
                latlon = f"{dataset.variables['lat'][4]}, {180 - dataset.variables['lon'][4]}"

                # title and legend
                plt.title(f"Time: {date}\nMedian Temp: {median}\nLatLon: {latlon}")
                # give title enough space
                plt.subplots_adjust(top=0.8)
                # plot
                plt.imshow(dataset.variables['tas'][idx, :, :])
                plt.show()


# join_climate_data("./data-files")


# create_train_datasets_from_era5("joined.nc", 48.13, 11.57)
# train("train_args.txt")

# coordinates hamburg
lat = 53.5
lon = 180 - 10

print(lat, lon)

create_usage_dataset_from_era5("joined.nc", lat, lon, [424, 2094])

evaluate("test_args.txt")

def plot_outputs(folder_path):
    # for each .nc file in folder matplotlib show
    for file in os.listdir(folder_path):
        if file.endswith(".nc"):
            with nc.Dataset(f"{folder_path}/{file}", 'r+') as dataset:
                for idx in range(len(dataset.variables['time'][:])):
                    # calculate human-readable date and time
                    # get step, and start date
                    hours = float(dataset.variables['time'][idx])
                    start_date = datetime.datetime.strptime(dataset.variables['time'].units,
                                                            "hours since %Y-%m-%d %H:%M:%S")
                    # add hours to start date
                    date = start_date + datetime.timedelta(hours=hours)
                    median = np.median(dataset.variables['tas'][idx, :, :])
                    latlon = f"{dataset.variables['lat'][4]}, {180 - dataset.variables['lon'][4]}"

                    # title and legend wth filename
                    plt.title(f"Time: {date}\nMedian Temp: {median}\nLatLon: {latlon}\nFile: {file}")
                    # give title enough space
                    plt.subplots_adjust(top=0.8)
                    # plot
                    plt.imshow(dataset.variables['tas'][idx, :, :])
                    plt.show()

plot_outputs("./outputs")
