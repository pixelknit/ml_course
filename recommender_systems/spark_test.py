from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("RecommenderSystem").getOrCreate()

data = spark.read.csv("simple_df.csv", header=True, inferSchema=True)
data = data.select(col("userId").cast("integer"),
    col("movieId").cast("integer"),
    col("rating").cast("float"))

(training, test) = data.randomSplit([0.8, 0.2])

als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

userRecs = model.recommendForAllUsers(10)
userRecs.show(5, truncate=False)

itemRecs = model.recommendForAllItems(10)
itemRecs.show(5, truncate=False)

spark.stop()
