"""
If the system is not hosted in a docker environment, the following file will be used to
create symlinks to provide the same functionality as the docker volumes.
"""

import os

def running_in_docker():
    return os.path.exists('/.dockerenv') or \
           os.path.isfile('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read()


# docker-compose.yml states in the volume section the following mapping:
# we ignore the requirements.txt

docker_volumes_string = """
    - ../climatereconstructionai:/climatereconstructionai
    - ./:/opt/airflow
    - /work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/1hr/atmos/tas/r1i1p1:/volumes/era5_hourly_tas
    - /work/bm1159/XCES/xces-work/k203179/data:/storage
    - /work/bm1159/XCES/xces-work/k203179/models:/models
"""

def create_symlinks_if_not_running_in_docker():

    if not running_in_docker():

        folder_mapping = docker_volumes_string.split("\n")
        folder_mapping = [x.replace('-', '') for x in folder_mapping if x]
        folder_mapping = [x.strip() for x in folder_mapping]

        for mapping in folder_mapping:
            source, destination = mapping.split(":")

            # calculate absolute path
            source = os.path.join(os.path.abspath(os.path.dirname(__file__)), source)
            destination = os.path.join(os.path.abspath(os.path.dirname(__file__)), destination)
            
            if not os.path.exists(source):
                # create source folder
                os.makedirs(source)
                print(f"Created source folder {source}")
                
            if not os.path.exists(destination):
                os.symlink(source, destination)
                os.chmod(destination, 0o777)
                print(f"Created symlink from {source} to {destination}")
            else:
                print(f"Symlink already exists from {source} to {destination}")