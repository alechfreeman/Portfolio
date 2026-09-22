import wget 
from pyspark import SparkContext
from itertools import combinations
import random
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FriendRec").getOrCreate()
sc = spark.sparkContext



file = wget.download("https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt")
data = sc.textFile(file)

#Pass in line and get the list of possible mutual friends of the form ((idA,idB), 1) 
def mapFunc(line):
    parts = line.split("\t")
    userID = parts[0]

    if len(parts) == 1:
        return []

    friends = parts[1].split(",")

    results = []

    #account for existing friendships
    for f in friends:
        results.append(((userID,f), -1))
    
    # get mutual friend pairs
    for f1, f2 in combinations(friends, 2):
        results.append(((f1, f2), 1))
        results.append(((f2, f1), 1))
    
    return results

#flatten lists returned from each line to form one large mapped dataset
mapped = data.flatMap(mapFunc)

# Convert to form ((1, 2), [1, 1, 1]), ((2, 67), [1, 1, -1])
reduced = mapped.groupByKey().mapValues(list)

def filterPair(pair):
    users, counts = pair
    if -1 in counts:
        return None
    return (users[0], (users[1], sum(counts)))

filtered = reduced.map(filterPair).filter(lambda x: x is not None)
#Now have (1, (2, 3), ... (firstNum, (pairedNumber, # of mutual friends))


grouped = filtered.groupByKey()
# Now have (1, [(2,3), (3, 4)]), ... = (user1, [(user i, # of mutual friends with user i)])

#get top 10 pairs in terms of # of mutual friends
def top10(recList):
    return sorted(recList, key = lambda x: -x[1])[:10]

recommendations = grouped.mapValues(top10)
#Have up to top 10 pairs 

#Pick random 10 users
sample_users = recommendations.takeSample(False, 10, seed=42)

#Print user and recommendations for the randomly sampled users.
for user, recs in sample_users:
    ids = [r[0] for r in recs]
    print(f"{user}\t{','.join(ids)}")



# PART 2

def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) != 2:
        return []
    user = parts[0]
    friends = parts[1].split(",")
    #Add the user, friend pairs to RDD and the rating of 1 if they occur
    return [(user, f, 1.0) for f in friends]

#Get the (user, friend, 1) tuples in ratings
ratings = data.flatMap(parse_line)

#Convert RDD to DF
ratings_df = ratings.toDF(["user", "item", "rating"])

user_indexer = StringIndexer(inputCol="user", outputCol="userIndex", handleInvalid="keep")
item_indexer = StringIndexer(inputCol="item", outputCol="itemIndex", handleInvalid="keep")


#ALS
als = ALS(userCol="userIndex", itemCol="itemIndex", ratingCol="rating",
            coldStartStrategy="drop", nonnegative=True)

#Pipeline
pipeline=Pipeline(stages=[user_indexer, item_indexer, als])

#Parameters to test
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1]) \
    .addGrid(als.maxIter, [5, 10]) \
    .build()

evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="rating", 
    predictionCol="prediction"
)

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

#Run cross validation and choose the best model
cv_model = crossval.fit(ratings_df)
best_model = cv_model.bestModel

#Extract ALS model of the last stage
als_model = best_model.stages[-1]

#Generate recommendations using the best model
user_recommendations = als_model.recommendForAllUsers(50)

#Convert index back to original ID for users and items/friends
user_labels = best_model.stages[0].labels
item_labels = best_model.stages[1].labels


#get existing friends
existing_friends = ratings.map(lambda x: (x[0], x[1])) \
                          .groupByKey() \
                          .mapValues(set) \
                          .collectAsMap()

existing_friends_bc = sc.broadcast(existing_friends)

#Convert rows indexes for users and items back to original IDs
def convert_recommendations(row):
    user = user_labels[int(row.userIndex)]
    recs = []
    existing = existing_friends_bc.value.get(user, set())
    for r in row.recommendations:
        candidate = item_labels[int(r.itemIndex)]
        if candidate != user and candidate not in existing:
            recs.append(candidate)
    return (user, recs[:10])

recommendations_rdd = user_recommendations.rdd.map(convert_recommendations)

recommendations_dict = dict(recommendations_rdd.collect())

sample_user_ids = [user for user, _ in sample_users]

for user in sample_user_ids:
    recs = recommendations_dict.get(user, [])
    print(f"{user}\t{','.join(recs)}")





