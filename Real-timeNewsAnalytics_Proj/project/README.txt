


0. Download elasticsearch, logstash, and kibana and appropriate libraries (newsapi-python, kafka-python)
1. Create NewsAPI account https://newsapi.org/docs/client-libraries/python
2. Get your personal API key from NewsAPI and set API_KEY constant in wordCount.py to that value
3. Set q in line 25 of wordCount.py to the topic you are interested in searching for
4. Start the kafka server according to the instructions on https://kafka.apache.org/quickstart/
5. Create the topics named topic1 and topic2
6. Start elasticsearch
    a. Start it (./elasticsearch-9.3.3/bin/elasticsearch)
    b. Check http://localhost:9200 works
    c. Login with user = elastic and password given from startup
7. Start kibana
    a. Obtain the token for kibana using command ./elasticsearch-9.3.3/bin/elasticsearch-create-enrollment-token -s kibana in WSL
    b. start kibana (./bin/kibana)
    c. Open https://localhost:5601
    d. login using user = elastic and the elasticsearch password
8. Start Logstash
    a. Configure the given logstash.conf file to include your password in the given logstash.conf file
    b. Start logstash ./logstash-9.3.3/bin/logstash -f logstash.conf
9. Run Spark NERStreaming.py
    a. Run spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 BD-Assignment3/NERStreaming.py
    - This reads from kafka topic1, extracts named entities, counts entities and outputs to topic2
10. Run Python producer
    a. Run python3 wordCount.py
    - This fetches news every 15 seconds and sends articles to topic1
11. Create Data View
    a. Go to Kibana or https://localhost:5601
    b. Set index pattern to entities*
12. Create Bar Chart
    a. Go to Visualize Library -> Create visualization -> Bar Chart
    b. on Y-axis, make the field count and aggregation the sum
    c. on X-axis, make the fiedl entity.keyword and choose Top 10 in descending order sorted by count sum
13. Use Kibana time filter and look at last 15, 30, 45, and 60 minutes 
