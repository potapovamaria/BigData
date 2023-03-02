"""
Analytics module for dataset
"""
from datetime import datetime

import pandas as pd  # type: ignore

import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

from  backend.preprocessing_data import preprocessing


def visualization(data_frame):
    """
    Data visualization as a graphic
    data_frame is dataset from DB
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_frame.index, y=data_frame.PAY, mode='lines'))

    fig.show()


def most_payment_day_of_week(data_frame):
    """
    Getting most payment day of week during all the period
    data_frame is dataset from DB
    return sum of points for every day of weeks
    """
    data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
    data_frame = data_frame.sort_values(by='Date')
    data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
    data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)
    dff = data_frame
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    values = data_frame[idx].WeekDay.values
    values_new = []
    for i in range(7):
        count = 0
        for value in enumerate(values):
            if value == i:
                count += 1
        values_new.append(count)

    return values_new
    # fig = px.histogram(df[idx].WeekDay.values)
    # fig.show()


def most_payment_day(data_frame):
    """
    Getting most payment day of a month during all the period
    data_frame is dataset from DB
    return sum of points for every day of months
    """

    data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
    data_frame = data_frame.sort_values(by='Date')
    data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
    data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)
    dff = data_frame
    dff['Day'] = dff.index.day
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    values = data_frame[idx].Day.values
    values_new = []
    for i in range(1, 31):
        count = 0
        for value in enumerate(values):
            if value == i:
                count += 1
        values_new.append(count)

    return values_new


def decomposition(data_frame):
    """
    Making decomposition of a data
    :param data_frame: dataset from DB
    :return: decomposition of a dataset
    """
    data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
    data_frame = data_frame.sort_values(by='Date')
    data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
    data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)
    decompos = seasonal_decompose(data_frame.PAY, model='multiplicative')

    trend = decompos.trend[~decompos.trend.isnull()][-200:]
    trend.index = trend.index.strftime('%d.%m.%Y')
    # trend = json.dumps(trend.to_dict())
    trend = trend.to_dict()
    seasonal = decompos.seasonal[~decompos.seasonal.isnull()][-200:]

    seasonal.index = seasonal.index.strftime('%d.%m.%Y')
    # seasonal = json.dumps(seasonal.to_dict())
    seasonal = seasonal.to_dict()
    resid = decompos.resid[~decompos.resid.isnull()][-200:]

    resid.index = resid.index.strftime('%d.%m.%Y')
    resid = resid.to_dict()
    # resid = json.dumps(resid.to_dict())

    return trend, seasonal, resid


def check_stationarity(data_frame):
    """
    Checking stationary of a time-series
    :param data_frame: dataset from a DB
    """
    adf = sm.tsa.adfuller(data_frame.PAY)
    p_value = adf[1]
    print('p-value: ', p_value)
    print('adf: ', adf[0])
    print('Critical values: ', adf[4])
    if adf[0] > adf[4]['5%']:
        print('Главный ряд не стационарен.')
    else:
        print('Главный ряд стационарен.')


def get_price(data_frame):
    """
    Making averaging by tariffs
    :param data_frame: dataset from a DB
    :return: new averaging dataset by tariffs
    """
    data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
    data_frame = data_frame.sort_values(by='Date')
    data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
    data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)
    dff = data_frame.copy()
    for i in range(len(dff)):
        if datetime.strptime("2011-05-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2012-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 2.8
        if datetime.strptime("2012-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2013-01-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 2.97
        if datetime.strptime("2013-01-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2014-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.39
        if datetime.strptime("2014-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2015-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.53
        if datetime.strptime("2015-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2016-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 3.84
        if datetime.strptime("2016-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2017-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.12
        if datetime.strptime("2017-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2018-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.32
        if datetime.strptime("2018-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2019-01-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.43
        if datetime.strptime("2019-01-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2019-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.61
        if datetime.strptime("2019-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2020-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.65
        if datetime.strptime("2020-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2021-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.82
        if datetime.strptime("2021-07-01", "%Y-%m-%d") <= dff.index[i] < \
                datetime.strptime("2022-07-01", "%Y-%m-%d"):
            dff['PAY'][i] = dff['PAY'][i] * 3.53 / 4.98

    return dff


if __name__ == '__main__':
    df = preprocessing()
    df = df.toPandas()
    data = get_price(df)
    # data = {'trend' : trend, 'seasonal' : seasonal, 'resid' : resid}
    # json_data = json.dumps(data)
    print(data)
