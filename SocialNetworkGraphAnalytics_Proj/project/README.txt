

1. Make sure all necessary packages are downloaded: pandas, wget, GraphFrames 
2. In WSL, run spark-submit   --driver-memory 6g   --executor-memory 6g   --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12   SocialNetworkAnalysis.py
3. Results are in the GF_output.txt file in the current directory