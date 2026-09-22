
from kafka import KafkaProducer
import json, time
from newsapi import NewsApiClient

API_KEY = "08555b3a614a4b43967143f7e0b4901d"

newsapi = NewsApiClient(api_key=API_KEY)

#create kafka producer which sends data from Python application into Kafka
#client that connects to Kafka and knows how to send messages in json format
producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8') #convert python dict into json string and convert string into bytes for kafka
)

seen_titles = set()


while True:
    print("Loop started")

    print("Calling NewsAPI...")
    articles = newsapi.get_everything(
    q='Iran',
    language='en',
    sort_by='publishedAt',
    page_size=50
    )

    print("API response received")

    for article in articles['articles']:
        print("Fetching articles...")
        title = article['title']
        description = article['description']

        text = (title or "") + " " + (description or "")

        if title and title not in seen_titles:
            seen_titles.add(title)
            print("Sending:", title)
            producer.send("topic1", {"text": text})
    producer.flush()


    time.sleep(15)

