from airflow.models import BaseOperator

class LocateStation(BaseOperator):
    
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