from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sklearn.metrics as metr
import statsmodels.tsa.stattools as st
from sklearn import metrics


def prepare_data(data, start_date=datetime(2018, 5, 2, 22, 17, 9), int_length=1,
                 train_int=20, test_int=5, shift=1):
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values("date", inplace=True)
    data = data[(data['date'] > start_date)]
    if len(data) < (train_int + test_int) * int_length:
        raise Exception
    ready_data = pd.DataFrame(columns=['count', 'sentiment', 'rate', 'date'])

    iteration = 0
    while iteration <= (test_int + train_int + shift + 1):
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
    return ready_data


def generate_shifts(data, shift=1):
    for i in range(0, shift):
        data['sentiment' + str(i + 1)] = data['sentiment'].shift(i + 1)
        data['count' + str(i + 1)] = data['count'].shift(i + 1)
    data.dropna(inplace=True)
    return data


def normalize(data, shift=1):
    data['prev_rate'] = data['rate'].shift(-1)
    # data['prev_sentiment'] = data['sentiment'].shift(-shift)
    # print(data)
    data['rate'] = data['rate'].diff()
    max_rate = data['rate'].max()
    min_rate = data['rate'].min()
    data['rate'] = (data['rate'] - min_rate) / (max_rate - min_rate)
    data['prev_rate'] = data['prev_rate'].diff()
    data['prev_rate'] = (data['prev_rate'] - min_rate) / (max_rate - min_rate)

    # data['sentiment'] = data['sentiment'].diff()
    # data['count'] = data['count'].diff()
    data['count'] = pd.to_numeric(data['count'])
    max_count = data['count'].max()
    min_count = data['count'].min()
    data['count'] = (data['count'] - min_count) / (max_count - min_count)
    for i in range(0, shift):
        data['count' + str(i + 1)] = (data['count' + str(i + 1)] - min_count) / (max_count - min_count)
    data['const'] = 1
    data.dropna(inplace=True)
    data.index = data.index - min(data.index)
    return data, min_rate, max_rate


def generate_sarimax_exog(data, shift=1):
    col_names = []
    for i in range(0, shift):
        col_names.append("sentiment" + str(i + 1))
        col_names.append("count" + str(i + 1))
    temp_data = data[col_names]
    temp_data.index = temp_data.index - min(temp_data.index.values)
    return temp_data


def calc(data, start_date=datetime(2018, 5, 2, 22, 17, 9), int_length=1,
         train_int=20, test_int=5, eps=5, shift=1):
    ready_data = prepare_data(data, start_date=start_date, int_length=int_length,
                              test_int=test_int, train_int=train_int, shift=shift)
    ready_data = generate_shifts(ready_data, shift=shift)
    ready_data, min_rate, max_rate = normalize(ready_data, shift=shift)
    data_train = ready_data[:train_int]
    data_test = ready_data[train_int:train_int + test_int]
    model1 = sm.OLS(data_train['rate'], np.matrix(generate_sarimax_exog(data_train, shift=shift), dtype='float'))
    results1 = model1.fit()
    print(results1.summary())
    data_test = data_test.dropna()
    pred = results1.predict(np.matrix(generate_sarimax_exog(data_test, shift=shift), dtype='float'))
    # print("MSE ", metrics.mean_squared_error(data_test['rate'], pred))

    # print(np.matrix(data_train['rate'].values, dtype='float'))
    # print(np.matrix(generate_sarimax_exog(data_train, shift=shift).values, dtype='float'))

    model2 = sm.tsa.statespace.SARIMAX(endog=np.array(data_train['rate'].values, dtype='float'),
                                       exog=np.matrix(generate_sarimax_exog(data_train, shift=shift).values,
                                                      dtype='float'),
                                       order=(shift, 0, 0),
                                       enforce_invertibility=False, enforce_stationarity=False)
    results2 = model2.fit()
    print(results2.summary())
    forecasts = results2.forecast(test_int,
                                  exog=np.matrix(generate_sarimax_exog(data_test, shift=shift).values, dtype='float'))
    # print(forecasts)

    # success_num = 0
    # for i in range(1, test_int):
    #     if abs(pred[train_int + i]) > eps:
    #         if pred[train_int + i] * data_test.iloc[i]['rate'] > 0:
    #             success_num += 1
    #     elif abs(data_test.iloc[i]['rate']) < eps:
    #         success_num += 1
    # success_rate = success_num / test_int
    # print('success_rate', success_rate)
    fig, ax2 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('sentiment', color=color)
    # ax1.plot_date(data_test["date"], data_test['sentiment'], color=color, linestyle="solid")
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    unnorm_rate = (data_test['rate'] * (max_rate - min_rate) + min_rate).values
    print(unnorm_rate)
    unnorm_pred = pred * (max_rate - min_rate) + min_rate
    print(unnorm_pred)
    unnorm_forecasts = forecasts * (max_rate - min_rate) + min_rate
    print(unnorm_forecasts)

    success_num = 0
    for i in range(0, test_int):
        if abs(unnorm_pred[i]) > eps:
            if unnorm_pred[i] * unnorm_rate[i] > 0:
                success_num += 1
        elif abs(unnorm_rate[i]) < eps:
            success_num += 1
    success_rate = success_num / test_int
    print('success_rate without prev', success_rate)

    print("MSE without prev: " + repr(metr.mean_squared_error(unnorm_rate, unnorm_pred)))
    print("RMSE without prev: " + repr(metr.mean_squared_error(unnorm_rate, unnorm_pred) ** 0.5))
    print("MAE without prev: " + repr(metr.mean_absolute_error(unnorm_rate, unnorm_pred)))
    print("R^2 without prev: " + repr(metr.r2_score(unnorm_rate, unnorm_pred)))

    success_num = 0
    for i in range(0, test_int):
        print("fcst: ", unnorm_forecasts[i])
        print("rate: ", unnorm_rate[i])
        if abs(unnorm_forecasts[i]) > eps:
            if (unnorm_forecasts[i] * unnorm_rate[i]) > 0:
                success_num += 1
                print("success")
        elif abs(unnorm_rate[i]) < eps:
            success_num += 1
            print("success")
    success_rate = success_num / test_int
    print('success_rate with prev', success_rate)

    print("MSE with prev: " + repr(metr.mean_squared_error(unnorm_rate, unnorm_forecasts)))
    print("RMSE with prev: " + repr(metr.mean_squared_error(unnorm_rate, unnorm_forecasts) ** 0.5))
    print("MAE with prev: " + repr(metr.mean_absolute_error(unnorm_rate, unnorm_forecasts)))
    print("R^2 with prev: " + repr(metr.r2_score(unnorm_rate, unnorm_forecasts)))

    color = 'tab:blue'
    ax2.set_ylabel('rate', color=color)  # we already handled the x-label with ax1
    ax2.plot_date(data_test["date"], unnorm_rate, color=color, linestyle="solid")
    ax2.set_ylim([-20, 20])
    # ax2.plot_date(data_test["date"], unnorm_pred, color='tab:green', linestyle="solid")
    ax2.plot_date(data_test["date"], unnorm_forecasts, color='tab:orange', linestyle="solid")
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
start = datetime(2018, 5, 2, 22, 23, 9)
int_length = 5
for i in range(0, 20):
    df = pd.read_csv("data.csv")
    calc(df, start_date=start, train_int=200, int_length=int_length, eps=5, shift=1, test_int=10)
    start = start + timedelta(minutes=int_length)
