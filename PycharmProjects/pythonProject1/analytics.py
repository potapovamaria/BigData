from datetime import datetime, date, timedelta
import json

import pandas as pd
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import plotly.express as px

import preprocessing_data


def visualisation(df):
    fig = go.Figure()  # создание графика
    fig.add_trace(go.Scatter(x=df.index, y=df.PAY,
                             mode='lines'))  # добавление графика с точками, соединенными линиями, где x - даты, а y - сумма оплат
    fig.show()


def most_payment_day_of_week(df):
    df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
    df = df.sort_values(by='Date')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date', 'PAY_DATE'], axis=1)
    dff = df
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    values = df[idx].WeekDay.values
    values_new = []
    for i in range(7):
        count = 0
        for j in range(len(values)):
            if values[j] == i:
                count += 1
        values_new.append(count)

    return values_new
    # fig = px.histogram(df[idx].WeekDay.values)
    # fig.show()


def most_payment_day(df):
    df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
    df = df.sort_values(by='Date')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date', 'PAY_DATE'], axis=1)
    dff = df
    dff['Day'] = dff.index.day
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    values = df[idx].Day.values
    values_new = []
    for i in range(31):
        count = 0
        for j in range(len(values)):
            if values[j] == i:
                count += 1
        values_new.append(count)

    return values_new

def decomposition(df):
    df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
    df = df.sort_values(by='Date')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date', 'PAY_DATE'], axis=1)
    decomposition = seasonal_decompose(df.PAY, model='multiplicative')

    trend = decomposition.trend[~decomposition.trend.isnull()][-200:]
    trend.index = trend.index.strftime('%d.%m.%Y')
    # trend = json.dumps(trend.to_dict())
    trend = trend.to_dict()
    seasonal = decomposition.seasonal[~decomposition.seasonal.isnull()][-200:]

    seasonal.index = seasonal.index.strftime('%d.%m.%Y')
    # seasonal = json.dumps(seasonal.to_dict())
    seasonal = seasonal.to_dict()
    resid = decomposition.resid[~decomposition.resid.isnull()][-200:]

    resid.index = resid.index.strftime('%d.%m.%Y')
    resid = resid.to_dict()
    # resid = json.dumps(resid.to_dict())

    return trend, seasonal, resid
    # fig = make_subplots(rows=3, cols=1)
    #
    # fig.add_trace(go.Scatter(y=decomposition.trend[~decomposition.trend.isnull()], mode="lines"), row=1, col=1)
    # fig.add_trace(go.Scatter(y=decomposition.seasonal[~decomposition.seasonal.isnull()][:200], mode="lines"), row=2, col=1)
    # fig.add_trace(go.Scatter(y=decomposition.resid[~decomposition.resid.isnull()], mode="lines"), row=3, col=1)
    # fig.show()

def check_stationarity(df):
    # Тест Дикки-Фуллера
    adf = sm.tsa.adfuller(df.PAY)
    p_value = adf[1]
    print('p-value: ', p_value)  # вероятность того, что нулевая гипотеза не будет отклонена
    print('adf: ', adf[0])  # критическое значение для ряда
    print('Critical values: ', adf[4])  # критические значения самого теста
    if adf[0] > adf[4][
        '5%']:  # тест Дикки-Фуллера: полученное значение должно быть меньше критического для стационарности ряда
        print('Главный ряд не стационарен.')
    else:
        print('Главный ряд стационарен.')

def get_price(df):
    df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
    df = df.sort_values(by='Date')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df = df.drop(['Date', 'PAY_DATE'], axis=1)
    dff = df.copy()
    for i in range(len(dff)):
        if dff.index[i] >= datetime.strptime("2011-05-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2012-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 2.8
        if dff.index[i] >= datetime.strptime("2012-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2013-01-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 2.97
        if dff.index[i] >= datetime.strptime("2013-01-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2014-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.39
        if dff.index[i] >= datetime.strptime("2014-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2015-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.53
        if dff.index[i] >= datetime.strptime("2015-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2016-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.84
        if dff.index[i] >= datetime.strptime("2016-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2017-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.12
        if dff.index[i] >= datetime.strptime("2017-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2018-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.32
        if dff.index[i] >= datetime.strptime("2018-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2019-01-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.43
        if dff.index[i] >= datetime.strptime("2019-01-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2019-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.61
        if dff.index[i] >= datetime.strptime("2019-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2020-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.65
        if dff.index[i] >= datetime.strptime("2020-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2021-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.82
        if dff.index[i] >= datetime.strptime("2021-07-01", "%Y-%m-%d") and dff.index[i] < datetime.strptime("2022-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.98

    return dff

if __name__ == '__main__':
    df = preprocessing_data.preprocessing()
    df = df.toPandas()
    data = get_price(df)
    # data = {'trend' : trend, 'seasonal' : seasonal, 'resid' : resid}
    # json_data = json.dumps(data)
    print(data)

