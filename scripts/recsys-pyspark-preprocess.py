import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME", "OUTPUT_PATH", "RATINGS_PATH",
                                      "EXISTING_USERS_MAP_PATH", "EXISTING_ITEMS_MAP_PATH",
                                      "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
                                      "POSTGRES_USER", "POSTGRES_PASSWORD"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

OUTPUT_PATH = args["OUTPUT_PATH"]
RATINGS_PATH = args["RATINGS_PATH"]
EXISTING_USERS_MAP = args["EXISTING_USERS_MAP_PATH"]
EXISTING_ITEMS_MAP = args["EXISTING_ITEMS_MAP_PATH"]

POSTGRES_HOST = args["POSTGRES_HOST"]
POSTGRES_PORT = args["POSTGRES_PORT"]
POSTGRES_DB = args["POSTGRES_DB"]
POSTGRES_USER = args["POSTGRES_USER"]
POSTGRES_PASSWORD = args["POSTGRES_PASSWORD"]

MIN_RATING = 4.0
MIN_INTERACTIONS = 5

rating = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)

pos_ratings = rating.filter(F.col("rating") >= F.lit(MIN_RATING)).withColumn("interaction", F.lit(1))
pos_ratings = pos_ratings.withColumn(
    "timestamp",
    F.when(
        F.col("timestamp").cast("long").isNotNull(),
        F.col("timestamp").cast("long")
    ).otherwise(F.unix_timestamp(F.col("timestamp")))
)

pos_ratings = pos_ratings.dropna(subset=["userId", "movieId", "timestamp"])

item_counts = (
    pos_ratings
    .groupBy("movieId")
    .agg(F.count("*").alias("item_interaction_count"))
    .filter(F.col("item_interaction_count") >= MIN_INTERACTIONS)
    .select("movieId")
)

user_counts = (
    pos_ratings
    .groupBy("userId")
    .agg(F.count("*").alias("user_interaction_count"))
    .filter(F.col("user_interaction_count") >= MIN_INTERACTIONS)
    .select("userId")
)

filtered_interactions = (
    pos_ratings
    .join(item_counts, on="movieId", how="inner")
    .join(user_counts, on="userId", how="inner")
    .dropDuplicates(["userId", "movieId", "timestamp"])
).cache()

def load_existing_map(path, id_col, idx_col):
    if path and path.lower() != "none":
        try:
            return spark.read.csv(path, header=True, inferSchema=True).select(id_col, idx_col)
        except Exception:
            pass
    return spark.createDataFrame([], f"{id_col} INT, {idx_col} INT")

existing_users_map = load_existing_map(EXISTING_USERS_MAP, "userId",  "user_idx")
existing_items_map = load_existing_map(EXISTING_ITEMS_MAP, "movieId", "item_idx")

max_user_idx = existing_users_map.agg(F.max("user_idx")).collect()[0][0]
max_item_idx = existing_items_map.agg(F.max("item_idx")).collect()[0][0]
max_user_idx = max_user_idx + 1 if max_user_idx is not None else 0
max_item_idx = max_item_idx + 1 if max_item_idx is not None else 0

all_users = filtered_interactions.select("userId").distinct()
new_users = all_users.join(existing_users_map, on="userId", how="left_anti")
user_window = Window.orderBy("userId")
new_users_map = new_users.withColumn(
    "user_idx", F.row_number().over(user_window) - 1 + max_user_idx
)
users_map = existing_users_map.union(new_users_map)

all_items = filtered_interactions.select("movieId").distinct()
new_items = all_items.join(existing_items_map, on="movieId", how="left_anti")
item_window = Window.orderBy("movieId")
new_items_map = new_items.withColumn(
    "item_idx", F.row_number().over(item_window) - 1 + max_item_idx
)
items_map = existing_items_map.union(new_items_map)

interactions_final = (
    filtered_interactions
    .join(users_map, on="userId", how="inner")
    .join(items_map, on="movieId", how="inner")
    .select(
        "userId",
        "movieId",
        "user_idx",
        "item_idx",
        "rating",
        "interaction",
        "timestamp",
    )
)

interactions_final.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/interactions")
users_map.select("userId", "user_idx").write.mode("overwrite").parquet(f"{OUTPUT_PATH}/users_map")
items_map.select("movieId", "item_idx").write.mode("overwrite").parquet(f"{OUTPUT_PATH}/items_map")

postgres_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
postgres_properties = {
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "driver": "org.postgresql.Driver"
}

movie_ids_list = [row.movieId for row in items_map.select("movieId").distinct().collect()]

if not movie_ids_list:
    postgres_movies = spark.createDataFrame(
        [], "movieId LONG, title STRING, genres STRING, tags STRING"
    )
else:
    movie_ids_str = ",".join(str(mid) for mid in movie_ids_list)
    postgres_items = spark.read.jdbc(
        url=postgres_url,
        table=f"(SELECT item_id, title, genre, tags FROM public.items WHERE item_id IN ({movie_ids_str})) AS items",
        properties=postgres_properties
    )
    postgres_movies = postgres_items.select(
        F.col("item_id").alias("movieId"),
        F.col("title"),
        F.col("genre").cast("string").alias("genres"),
        F.col("tags").cast("string").alias("tags")
    )

movies_df = postgres_movies.join(
    items_map,
    on="movieId",
    how="inner"
).select(
    "movieId",
    "item_idx",
    "title",
    "genres",
    "tags"
)

movies_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/movies")

tags_df = spark.createDataFrame([], "movieId LONG, userId LONG, tag STRING, timestamp TIMESTAMP")
tags_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/tags")

job.commit()
