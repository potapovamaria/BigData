"""
DB Initialization Module
"""
import os
from pandas import DataFrame
from pymongo import MongoClient

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jre1.8.0_351"
os.environ["SPARK_HOME"] = "C:\\Users\\potap\\PycharmProjects\\spark-3.3.1-bin-hadoop3"
os.environ["HADOOP_HOME"] = "C:\\Users\\potap\\PycharmProjects\\hadoop-3.0.0"

def init_data():
    """ Init database """
    mongo_client = MongoClient("mongodb://localhost:27017/")
    database = mongo_client['YOUR_DB_NAME']
    collection = database['your_collection']

    item_details = collection.find()

    items_df = DataFrame(item_details)

    return items_df

if __name__ == '__main__':
    init_data()
    print(init_data())
