import wget 
from pyspark import SparkContext
from itertools import combinations
import random
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import zipfile
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re


spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

file = wget.download("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip")

with zipfile.ZipFile(file, 'r') as zip_ref:
    zip_ref.extractall(".")

data = sc.textFile("SMSSpamCollection")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

#Preprocess the text
def preprocess(text):
    text = text.lower()
    words = re.findall(r'\w+', text)
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return words

dataset = data.map(lambda line: line.split("\t")).filter(lambda x: len(x) == 2).map(lambda x: (x[0], preprocess(x[1])))

train, test = dataset.randomSplit([0.8, 0.2], seed=42)

#MR Naive Bayes 

#Documents per class (PRIORS) calculation

class_counts = train.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
total_docs = train.count()
priors = class_counts.mapValues(lambda x / total_docs).collectAsMap()

#Probability of word given class => P(word | class) = Word count of word in class + 1 / Word Count of Class + Vocab Size (for Laplace smoothing)
word_counts = train.flatMap(lambda x: [((x[0], word), 1) for word in x[1]]).reduceByKey(lambda a,b:a+b)

#Word count per class

total_words_per_class = word_counts.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda a,b: a + b).collectAsMap()


#Vocabulary size = amount of total distinct words
vocab_size = word_counts.map(lambda x: x[0][1]).distinct().count()

#Compute P(word | class) probabilities
word_probs = word_counts.map(lambda x: ((x[0][0], x[0][1]), (x[1] + 1) / (total_words_per_class[x[0][0]] + vocab_size ))).collectAsMap()

#Compute prediction
#Given a document or message, compute a score for each class = log P(class) + sum{i=1 -> n}(log P(wi|class))
def predict(doc):
    scores = {}
    sum = 0
    for c in priors:
        scores[c] = math.log(priors[c])
    
        for word in doc:
            if (c, word) in word_probs: #if word is already seen use already calculated P(word | class)
                scores[c] += math.log(word_probs[(c, word)])
            else: #word not seen -> recalculate 
                scores[c] += math.log(1/ (total_words_per_class[c] + vocab_size))
    return max(scores, key=scores.get)

#Get predictions for each test example
predictions = test.map(lambda x: (x[0], predict(x[1]))).collect()

y_true = [t[0] for t in predictions]
y_pred = [t[1] for t in predictions]

# Print metrics
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

precision = precision_score(y_true, y_pred, pos_label='spam')
recall = recall_score(y_true, y_pred, pos_label='spam')
f1 = f1_score(y_true, y_pred, pos_label='spam')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)














