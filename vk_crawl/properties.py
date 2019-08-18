from airflow import DAG, utils
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import configparser

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'src'))
import process_properties as properties

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.ini"))

default_args = {
    'owner': config['airflow']['owner'],
    'start_date': utils.dates.days_ago(1),
    'retries': 0
}

dag = DAG('vk_properties', default_args=default_args, catchup=False, schedule_interval=timedelta(minutes=5))
priorities1 = PythonOperator(
            task_id='calculate_properties_users',
            python_callable=properties.process_users,
            pool=config['airflow']['pool'],
            op_kwargs={},
            dag=dag)

priorities2 = PythonOperator(
            task_id='calculate_properties_groups',
            python_callable=properties.process_groups,
            pool=config['airflow']['pool'],
            op_kwargs={},
            dag=dag)
