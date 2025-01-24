from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace

def preprocess_spark(file_path):
    spark = SparkSession.builder.appName("AmazonReviewPreprocessing").getOrCreate()
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Text cleaning
    df = df.withColumn("cleaned_text", lower(col("reviewText")))
    df = df.withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"[^a-zA-Z\s]", ""))
    
    df.write.csv("../data/processed_data/spark_processed_reviews.csv", header=True)
    spark.stop()

if __name__ == "__main__":
    preprocess_spark("../data/sample_reviews.csv")
