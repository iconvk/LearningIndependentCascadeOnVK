import configparser
from airflow import DAG, utils
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'src'))
import process_posts as ps


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.ini"))

default_args = {
    'owner': config['airflow']['owner'],
    'start_date': utils.dates.days_ago(1),
    'retries': 0
}

dag = DAG('vk_posts', default_args=default_args, catchup=False, schedule_interval=timedelta(minutes=10))
priorities = PythonOperator(
            task_id='calculate_posts',
            python_callable=ps.process_nodes,
            pool=config['airflow']['pool'],
            op_kwargs={},
            dag=dag)
