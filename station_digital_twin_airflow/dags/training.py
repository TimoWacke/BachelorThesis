from config.symlink import create_symlinks_if_not_running_in_docker
import os

from datetime import datetime, timedelta
from airflow import DAG

from operators.locate_station import LocateStation
from operators.csv_to_nc import CSVtoNC

create_symlinks_if_not_running_in_docker()

ARGS = {
    "owner": "Wacke",
    "start_date": datetime(2021, 1, 1),
    "retries": 1,
    "email": ["wacke@dkrz.de"],
    "email_on_failure": True,
    "email_on_retry": False,
}

DAG_DOC_MD = """
    TODO: Add description
"""

with DAG(
        dag_id="basic_training",
        default_args=ARGS,
        schedule_interval="30 9 * * *",
        catchup=False,
        doc_md=DAG_DOC_MD,
) as dag:
    
        locate_station = LocateStation(
            task_id="locate_station",
            station_name="Hamburg",
        )
    
        csv_to_nc = CSVtoNC(
            task_id="csv_to_nc",
            station_name="Hamburg",
        )
    
        locate_station >> csv_to_nc