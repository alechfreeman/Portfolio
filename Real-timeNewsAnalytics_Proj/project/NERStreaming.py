
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType
import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import explode


spark = SparkSession.builder.appName("NERStreaming").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "topic1") \
            .load()


#The Json will have a field called text and it is a string
schema = StructType().add("text", StringType())

#Convert Kafka bytes into string -> '{"text": "..."}'
#Parse json into structured column data = -> {text: "..."}
#Extract the text field -> dataframe with "text" as column name
json_df = df.selectExpr("CAST(value AS STRING)") \
                .select(from_json(col("value"), schema).alias("data")) \
                .select("data.text")

#Load pre-trained NLP model from spacy to detect named entities
nlp = spacy.load("en_core_web_sm")

#Run model on text and return list of named entity strings
def extract_entities(text):
    doc = nlp(text)
    allowed_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "TIME", "QUANTITY"}
    return [ent.text for ent in doc.ents if ent.label_ in allowed_labels]

#wrap function in user defined function that can be applied to each row of the DF and returns a list of strings
ner_udf = udf(extract_entities, ArrayType(StringType()))

#For every row in DF run extract_entities(text) with the UDF and store the result in a new column "entities"
entities_df = json_df.withColumn("entities", ner_udf(col("text")))

#For every row in entities_df, create a separate row for each word in the list in the new dataframe
exploded = entities_df.select(explode(col("entities")).alias("entity"))

#Get the counts of each word/entity 
counts = exploded.groupBy("entity").count()

# counts.writeStream \
#         .outputMode("complete") \
#         .format("console") \
#         .start()

#Convert entity column to string use it as the Kafka message key foe easier downstream processing because all counts for a key stick together
#Take entire row and convert into json string which is the value: (entity="..", count=..) -> {"entity": "..", "count": 15} 
output = counts.selectExpr("CAST(entity AS STRING) AS key", 
                            "to_json(struct(*)) AS value")

#output DF is now | key | value | 

query = output.writeStream \
            .format("kafka")\
            .option("kafka.bootstrap.servers", "localhost:9092")\
            .option("topic", "topic2")\
            .option("checkpointLocation", "/tmp/checkpoints") \
            .outputMode("complete") \
            .start()

query.awaitTermination()





