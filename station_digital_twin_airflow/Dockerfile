# Use an Airflow base image
FROM apache/airflow:latest

# Install CDO (Climate Data Operators)
RUN apt-get update && \
    apt-get install -y cdo

# Install any additional dependencies
RUN pip install -r requirements.txt

# Copy your pipeline code into the container
COPY . /opt/airflow/dags/

# Set the working directory
WORKDIR /opt/airflow/dags/

# Start Airflow web server
CMD ["airflow", "webserver", "--port", "8080"]
