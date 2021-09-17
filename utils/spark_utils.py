import pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *


def get_local_spark_session():
    spark = get_spark_session()
    spark.conf.set("spark.driver.host", "localhost")
    # Setup hadoop fs configuration for schema gs://
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    conf.set("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    return spark


def get_spark_session():
    return SparkSession \
        .builder \
        .master("local[*]") \
        .appName("ProteinStructures") \
        .getOrCreate()


def mmcif_to_parquet(path, spark: SparkSession = None):
    sc = pyspark.SparkContext()
    files = spark.wholeTextFiles('gs://capstone-fall21-protein/UP000005640_9606_HUMAN/cif/*')


def load_protein_structures(path, limit: Int = None, spark: SparkSession = None):
    df = spark.read.csv(path)
    return df if not limit else df.take(limit)


def to_sequences(protein_structure_table: DataFrame):
    protein_sequences = protein_structure_table.groupby()
