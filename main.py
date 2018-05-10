from datetime import datetime, timedelta

import configparser
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

from sklearn import preprocessing


def get_db(connection_string, db_name):
    client = MongoClient(connection_string)
    db = client[db_name]
    return db


def get_rates(col, symbol):
    rates = col.find({"symbol": symbol})
    return rates


def get_rates_dated(col, symbol, start_date, end_date):
    rates = col.find({"symbol": symbol, "date": {"$lt": end_date, "$gt": start_date}})
    return rates


def get_posts(col):
    return col.count()


def get_avg_sentiment(col, start_date, end_date):
    agg_string = [{"$match": {"createdAt": {"$lte": end_date, "$gte": start_date}}},
                  {"$group": {"_id": None, "avg": {"$avg": "$compound"}, "count": {"$sum": 1}}}]
    cursor = col.aggregate(agg_string)
    return cursor


def get_tweets(col, start_date, end_date):
    return col.find({"createdAt": {"$lt": end_date, "$gt": start_date}})


config = configparser.ConfigParser()
config.read("mongo_config.ini")
connection_string = config["mongo"]["connection_string"]
db = get_db(connection_string, 'test')
col_rates = db.rate
# rates = get_rates(col_rates, 'BTCUSDT')
# for rate in rates:
#      print(rate)
start = datetime(2018, 5, 5, 22, 17, 9)
end = datetime(2018, 5, 6, 22, 17, 9)

delta = timedelta(minutes=10)
col_tweets = db.tweets

# while start < end:
#     tweets_by_date = get_avg_sentiment(col_tweets, start, start + delta)
#     print(start + delta)
#     for tweet in tweets_by_date:
#         print(tweet)
#     start = start + delta

rates_cursor = get_rates_dated(col_rates, "BTCUSDT", start, end)
rates_df = pd.DataFrame(list(rates_cursor))
rates_df = rates_df.sort_values("date")

avg_list = list()
count_list = list()
for i in range(1, len(rates_df)):
    print(rates_df.iloc[i - 1]["date"])
    tweets_cursor = get_avg_sentiment(col_tweets, rates_df.iloc[i - 1]["date"], rates_df.iloc[i]["date"])
    entry = tweets_cursor.next()
    avg = entry["avg"]
    avg_list.append(avg)
    num = entry["count"]
    count_list.append(num)
    tweets_df = pd.DataFrame(list(tweets_cursor))

rates_df.drop(rates_df.index[[len(rates_df) - 1]], inplace=True)
rates_df["sentiment"] = avg_list
rates_df["count"] = count_list
rates_df = rates_df[["date", "rate", "symbol", "sentiment", "count"]]
rates_df.to_csv("more_data.csv")
# num_values = rates_df[["sentiment", "rate"]].values
# min_max_scaler = preprocessing.MinMaxScaler()
# num_values_minmax = min_max_scaler.fit_transform(num_values)
# print(num_values)
# print(num_values_minmax)
# print(rates_df["sentiment"].corr(rates_df["rate"]))
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('sentiment', color=color)
# ax1.plot_date(rates_df["date"], rates_df['sentiment'], color=color, linestyle="solid")
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('rate', color=color)  # we already handled the x-label with ax1
# ax2.plot_date(rates_df["date"], rates_df['rate'], color=color, linestyle="solid")
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
