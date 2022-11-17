from plotly.subplots import make_subplots

import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import plotly.express as px


def visualisation(df):
    fig = go.Figure()  # создание графика
    fig.add_trace(go.Scatter(x=df.index, y=df.PAY,
                             mode='lines'))  # добавление графика с точками, соединенными линиями, где x - даты, а y - сумма оплат
    fig.show()


def most_payment_day_of_week(df):
    dff = df
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    return df[idx].WeekDay.values
    # fig = px.histogram(df[idx].WeekDay.values)
    # fig.show()


def most_payment_day(df):
    dff = df
    dff['Day'] = dff.index.day
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    idx = dff.groupby(['Month', 'Year'])['PAY'].transform(max) == dff['PAY']
    return df[idx].Day.values

def decomposition(df):
    decomposition = seasonal_decompose(df.PAY, model='multiplicative')  # КАК РАСКЛАДЫВАЕТСЯ???????????
    # decomposition.plot()  # 1 график = Тренд * Сезонность * Остаток
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(y=decomposition.trend[~decomposition.trend.isnull()], mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(y=decomposition.seasonal[~decomposition.seasonal.isnull()][:200], mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(y=decomposition.resid[~decomposition.resid.isnull()], mode="lines"), row=3, col=1)
    fig.show()

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

