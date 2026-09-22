

from pyspark.sql import SparkSession
import wget
from pyspark.sql.functions import split, col, desc
from graphframes import GraphFrame

spark = SparkSession.builder \
            .appName("GraphFrameE")\
            .getOrCreate()


wget.download("https://snap.stanford.edu/data/facebook_combined.txt.gz")

lines = spark.read.text("facebook_combined.txt.gz")

#Take single column of lines and splits it into two columns of the two vertices and returns it into new DF
#Select these computed columns and build new DF from them
edges = lines.select( 
        split(col("value"), " ").getItem(0).alias("src"),
        split(col("value"), " ").getItem(1).alias("dst")
)

#Dataset is undirected so make directed
reversed_edges = edges.select(
    col("dst").alias("src"),
    col("src").alias("dst")
)

edges = edges.union(reversed_edges)


#Get vertices as the distinct source and destinations 
#Use col for .alias 
vertices = edges.select(col("src").alias("id")) \
            .union(edges.select(col("dst").alias("id"))) \
            .distinct()

g = GraphFrame(vertices, edges)
g.cache()

#a. Find top 5 nodes with the highest outdegree and the find the count of the number of outgoing edges in each
outdeg = g.outDegrees.orderBy(desc("outDegree")).limit(5)

# b. Find the top 5 nodes with the highest indegree and find the count of the number of incoming edges in each
indeg = g.inDegrees.orderBy(desc("inDegree")).limit(5)

# c.  Calculate PageRank for each of the nodes and output the top 5 nodes with the highest PageRank values. You are free to define any suitable parameters

pageRank = g.pageRank(resetProbability=0.15, maxIter=10)
topPR = pageRank.vertices.orderBy(desc("pagerank")).limit(5)

#d. Run the connected components algorithm on it and find the top 5 components with the largest number of nodes.

spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
cc = g.connectedComponents()
topcc = cc.groupBy("component").count().orderBy(desc("count")).limit(5)


#e.  Run the triangle counts algorithm on each of the vertices and output the top 5 vertices with the largest triangle count. In case of ties, you can randomly select the top 5 vertices.

tc = g.triangleCount().orderBy(desc("count")).limit(5)

def df_to_string(df):
    return df.toPandas().to_string(index=False)

output = f"""
a. Top 5 Nodes by OutDegree
{df_to_string(outdeg)}

b. Top 5 Nodes by InDegree
{df_to_string(indeg)}

c. Top 5 Nodes by PageRank
{df_to_string(topPR.select("id", "pagerank"))}

d. Top 5 Largest Connected Components
{df_to_string(topcc)}

e. Top 5 Vertices by Triangle Count
{df_to_string(tc.select("id", "count"))}
"""
output_file = "GF_output.txt"
with open(output_file, "w") as f:
    f.write(output)


