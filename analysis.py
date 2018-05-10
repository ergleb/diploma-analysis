from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics


def calc(data, start_date=datetime(2018, 5, 2, 22, 17, 9), int_length=1,
         train_int=20, test_int=5, eps=5):
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values("date", inplace=True)
    data = data[(data['date'] > start_date)]
    if len(data) < (train_int + test_int) * int_length:
        raise Exception
    ready_data = pd.DataFrame(columns=['count', 'sentiment', 'rate', 'date'])

    iteration = 0
    while iteration <= (test_int + train_int):
        temp_data = data[iteration * int_length:(iteration + 1) * int_length]
        count = 0
        avg = 0.0
        for index, row in temp_data.iterrows():
            count += row['count']
            avg += row['sentiment']
        avg /= int_length
        result_dict = {
            'count': count,
            'sentiment': avg,
            'rate': data.iloc[(iteration + 1) * int_length - 1]['rate'],
            'date': data.iloc[(iteration + 1) * int_length - 1]['date']
        }
        ready_data = ready_data.append(pd.DataFrame(result_dict, index=[0]), ignore_index=True)
        iteration += 1

    ready_data['date'] = pd.to_datetime(ready_data['date'])
    data.sort_values("date", inplace=True)
    ready_data['prev_rate'] = ready_data['rate'].shift(-1)
    ready_data['rate'] = ready_data['rate'].diff()
    ready_data['prev_rate'] = ready_data['prev_rate'].diff()
    # ready_data['sentiment'] = ready_data['sentiment'].diff()
    # ready_data['count'] = ready_data['count'].diff()
    ready_data['count'] = pd.to_numeric(ready_data['count'])
    ready_data['const'] = 1
    data_train = ready_data[:train_int]
    data_test = ready_data[train_int:]
    model1 = sm.OLS(data_train['rate'].dropna(), data_train[['sentiment', 'count', 'const', 'prev_rate']].dropna())
    results1 = model1.fit()
    print(results1.summary())
    data_test = data_test.dropna()
    pred = results1.predict(data_test[['sentiment', 'count', 'const', 'prev_rate']].dropna())
    print(len(pred))
    print(len(data_test['rate']))
    print("MSE ", metrics.mean_squared_error(data_test['rate'], pred))
    success_num = 0
    for i in range(1, test_int):
        if abs(pred[train_int + i - 1]) > eps:
            if pred[train_int + i - 1] * data_test.iloc[i]['rate'] > 0:
                success_num += 1
        elif abs(data_test.iloc[i]['rate']) < eps:
            success_num += 1
    success_rate = success_num / test_int
    print('success_rate', success_rate)
    fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('sentiment', color=color)
    # ax1.plot_date(data["date"], data['sentiment'], color=color, linestyle="solid")
    # ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('rate', color=color)  # we already handled the x-label with ax1
    ax2.plot_date(data_test["date"], data_test['rate'], color=color, linestyle="solid")
    ax2.plot_date(data_test["date"], pred, color='tab:green', linestyle="solid")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# start = datetime(2018, 5, 3, 1, 22, 9)
# df = pd.read_csv("data.csv")
# calc(df, start_date=start, train_int=20, int_length=5, eps=5)
# start = datetime(2018, 5, 4, 12, 22, 9)
# df = pd.read_csv("data.csv")
# calc(df, start_date=start, train_int=20, int_length=15, eps=5)
# start = datetime(2018, 5, 3, 2, 22, 9)
# df = pd.read_csv("data.csv")
# calc(df, start_date=start, train_int=20, int_length=30, eps=5)
start = datetime(2018, 5, 2, 22, 22, 9)
for i in range(0,20):
    df = pd.read_csv("data.csv")
    calc(df, start_date=start, train_int=20, int_length=60, eps=5)
    start = start + timedelta(hours=1)
