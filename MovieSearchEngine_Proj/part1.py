import spacy
import wget
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("wordCounts")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

filename = wget.download("https://www.gutenberg.org/ebooks/2701.txt.utf-8")
input = sc.textFile(filename)


def extract_entities(partition):
    nlp = spacy.load("en_core_web_sm")
    allowed_labels = {"PERSON", "ORG", "GPE", "LOC"}
    for line in partition:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ in allowed_labels:
                yield ent.text

entities_rdd = input.mapPartitions(extract_entities)

sorted_results = entities_rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False)

print(sorted_results.take(50))
