
import wget
import shutil
import tarfile
import os
import re
import math
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

file = wget.download("http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz")


with tarfile.open(file, "r:gz") as tar:
    tar.extract("MovieSummaries/plot_summaries.txt")
    tar.extract("MovieSummaries/movie.metadata.tsv")
shutil.move("MovieSummaries/plot_summaries.txt", "plot_summaries.txt")
shutil.move("MovieSummaries/movie.metadata.tsv", "movie.metadata.tsv")

search_terms = sc.textFile("search_terms.txt")
metadata = sc.textFile("movie.metadata.tsv")

plots = sc.textFile("plot_summaries.txt")

stop_words = set(stopwords.words('english'))

def preprocess(line):
    parts = line.split("\t")
    if len(parts) != 2: 
        return None
    movie_id = parts[0]
    text = parts[1].lower()

    words = re.findall(r'\b[a-z]+\b', text)
    filtered = [w for w in words if w not in stop_words]

    return (movie_id, filtered)
    
processed = plots.map(preprocess).filter(lambda x: x is not None)

N_docs = processed.count()

tf = processed.flatMap(lambda x: [((x[0], word), 1) for word in x[1]]).reduceByKey(lambda x,y: x + y)

# ((docID, term), term_count)

df = processed.flatMap(lambda x: [(word, 1) for word in set (x[1])]).reduceByKey(lambda a, b: a + b)

#(term, docs_with_term)

idf = df.mapValues(lambda x: math.log(N_docs / x))
#(term, idf)

tfidf = tf.map(lambda x: (x[0][1], (x[0][0], x[1]))).join(idf)
#(term, ((docID, tf), idf))

F_tfidf = tfidf.map(lambda x: ((x[1][0][0], x[0]), x[1][0][1] * x[1][1]))

#((docID, term), tfidf)

search_list = search_terms.collect()

metadata = metadata.map(lambda line: line.split("\t")).map(lambda parts: (parts[0], parts[2]))

metadata_dict = dict(metadata.collect())

def search_single_term(term):
    results = F_tfidf.filter(lambda x: x[0][1]== term).map(lambda x: (x[0][0], x[1])).sortBy(lambda x: -x[1]).take(10)
    return [metadata_dict.get(doc_id, "Unknown") for doc_id, score in results]


#COSINE SIMILARITY
doc_norms = F_tfidf.map(lambda x: (x[0][0], x[1]**2)).reduceByKey(lambda a, b: a + b).mapValues(lambda x: math.sqrt(x))
# (docID, ||d||)

#idf dict
idf_dict = dict(idf.collect())

from collections import Counter

def search_multi_term(query):

    #preprocess query
    query_terms = query.lower().split()

    #compute query tf
    query_tf = Counter(query_terms)

    #{term : count, term: count}
    query_tfidf = {}
    # {term: tfidf, term : tfidf}
    for term, tf_val in query_tf.items():
        if term in idf_dict: 
            query_tfidf[term] = tf_val * idf_dict[term]

    if len(query_tfidf) == 0:
        return []

    query_terms_set = set(query_tfidf.keys())

    #compute dot products 
    dot_products = F_tfidf.filter(lambda x: x[0][1] in query_terms_set).map(lambda x: (x[0][0], x[1] * query_tfidf[x[0][1]])).reduceByKey(lambda a, b: a + b)

    # (docID, dotproduct)
    
    #query norm
    query_norm = math.sqrt(sum(x**2 for x in query_tfidf.values()))
    
    cosine_scores = dot_products.join(doc_norms).mapValues(lambda x: x[0] / (x[1] * query_norm))

    #(docID, cosine_score)
    

    top10 = cosine_scores.sortBy(lambda x: -x[1]).take(10)

    return [metadata_dict.get(doc_id, "Unknown") for doc_id, score in top10]


# read search file and run for each search item
for query in search_list:
    print("\n===============")
    print("Query:", query)

    if len(query.split()) == 1:
        print("Single-term query")
        results = search_single_term(query.lower())

    else:
        print("Multi-term query")
        results = search_multi_term(query)

    for movie in results:
        print(movie)



