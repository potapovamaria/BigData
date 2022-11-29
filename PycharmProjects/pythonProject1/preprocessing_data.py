import os
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as fun
from init_bd import init_data
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jre1.8.0_351"
os.environ["SPARK_HOME"] = "C:\\Users\\potap\\PycharmProjects\\spark-3.3.1-bin-hadoop3"
os.environ["HADOOP_HOME"] = "C:\\Users\\potap\\PycharmProjects\\hadoop-3.0.0"

def preprocessing():
    data = init_data()
    data = data.drop(['_id'], axis=1)
    findspark.init()

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Service_Spark") \
        .getOrCreate()

    df = spark.createDataFrame(data)

    dataset = df.select(col('PAY'),
                        col('PAY_DATE'))

    dataset = dataset.withColumn('PAY', fun.regexp_replace('PAY', ',', '.'))

    dataset = dataset.withColumn('PAY', dataset['PAY'].cast("float").alias('PAY'))

    dataset = dataset.withColumn('PAY', dataset['PAY'] / 1000)
    dataset = dataset.withColumn('PAY', dataset['PAY'].cast("float").alias('PAY'))
    dataset = dataset.withColumn('PAY_DATE', fun.to_date('PAY_DATE', format='dd.MM.yyyy'))
    dataset = dataset.withColumn('PAY_DATE', fun.date_format('PAY_DATE', format='dd.MM.yyyy'))
    dataset = dataset.withColumn('PAY_DATE', fun.to_date('PAY_DATE', format='dd.MM.yyyy'))
    print(dataset.printSchema())
    return dataset

if __name__ == '__main__':
    df = preprocessing()
    df.show(5)
