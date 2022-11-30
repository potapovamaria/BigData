import csv
import json
import pandas as pd
import sys, getopt, pprint
import os
from pandas import DataFrame
from pymongo import MongoClient

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jre1.8.0_351"
os.environ["SPARK_HOME"] = "C:\\Users\\potap\\PycharmProjects\\spark-3.3.1-bin-hadoop3"
os.environ["HADOOP_HOME"] = "C:\\Users\\potap\\PycharmProjects\\hadoop-3.0.0"

def init_data():
    # df = pd.read_csv('pay20220401.csv', sep=';')
    # df['PAY'] = df['PAY'].apply(lambda x: float(x.replace(',', '.')))
    # df['PAY'] = df['PAY'] / 1000
    # df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%d.%m.%Y')
    # df = df.sort_values(by='Date')
    # df = df.drop(df.columns[0], axis=1)
    # df = df.drop(['PAY_DATE', 'CNT'], axis=1)
    mongo_client = MongoClient("mongodb://localhost:27017/")
    database = mongo_client['YOUR_DB_NAME']
    collection = database['your_collection']

    item_details = collection.find()

    items_df = DataFrame(item_details)

    return items_df

if __name__ == '__main__':
    init_data()
    print(init_data())