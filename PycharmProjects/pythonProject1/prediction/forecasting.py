"""
Forecasting Module
"""

import warnings

import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn import svm



import xgboost
import lightgbm as lgb

from lssvr import LSSVR





from backend.preprocessing_data import preprocessing

warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    """ Calculate mape """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_prediction(df_new, prediction, start_date, pred_column):
    """ Plotting prediction and real data """
    plt.figure(figsize=(25, 12))
    plt.plot(df_new.index[df_new.index >= start_date],
             prediction, label="prediction", alpha=.7)
    plt.plot(df_new.index[df_new.index >= start_date],
             df_new.loc[df_new.index >= start_date, pred_column], label="real", alpha=.7)
    plt.scatter(df_new.index[df_new.index >= start_date],
                prediction, label="prediction", alpha=.7)
    plt.scatter(df_new.index[df_new.index >= start_date],
                df_new.loc[df_new.index >= start_date, pred_column], label="real", alpha=.7)
    plt.legend()
    plt.title(pred_column + " Prediction")
    plt.show()


def xgboost_model(train_x, train_y):
    """ Initialization of XGBoostModel """
    model = xgboost.XGBRegressor(booster='gblinear',
                                 learning_rate=0.2111,
                                 random_state=42,
                                 n_estimators=197)
    model.fit(train_x, train_y)

    return model


def lssvr_model(train_x, train_y):
    """ Initialization of LSSVRModel """
    lsclf = LSSVR(kernel='rbf',
                  C=10,
                  gamma=0.04)
    train_ = np.reshape(train_y, train_y.shape[0])
    lsclf.fit(train_x, train_)

    return lsclf


def svr_model(train_x, train_y):
    """ Initialization of SVRModel """
    clf = svm.SVR(kernel='rbf',
                  gamma=0.02,
                  C=2.32,
                  verbose=False,
                  tol=1e-10,
                  epsilon=0.0094)
    clf.fit(train_x, train_y)

    return clf


def lgbm_model(train_x, train_y):
    """ Initialization of LGBMModel """
    model_lgb = lgb.LGBMRegressor(learning_rate=0.071, max_depth=11, n_estimators=132,
                                  boosting_type='goss',
                                  num_leaves=25, random_state=42)
    model_lgb.fit(train_x, train_y)

    return model_lgb


def get_picks(dff, start_date, input_len, pred_len):
    """ Pick forecasting """
    # dff = pd.read_csv(file, sep=';')
    # dff['PAY'] = dff['PAY'].apply(lambda x: float(x.replace(',', '.')))
    # dff['PAY'] = dff['PAY'] / 1000
    # dff['Date'] = pd.to_datetime(dff['PAY_DATE'], format='%d.%m.%Y')
    print(dff)
    dff['Month'] = dff.index.month
    dff['Year'] = dff.index.year
    dff['WeekDay'] = dff.index.day_of_week
    dff = dff.sort_values(by='Date')

    # dff = dff.set_index(pd.DatetimeIndex(dff['Date']))

    df_peak = pd.DataFrame()
    grouped = dff.groupby(['Year', 'Month'])
    for group in grouped:
        perc = np.percentile(group.PAY, 95)
        df_peak = pd.concat([df_peak, group[group.PAY > perc]], axis=0)
    dfq = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', periods=len(dff), freq='D'),
                        'PAY': range(len(dff))})
    dfq.PAY = np.nan

    for i in df_peak.index:
        dfq.loc[dfq[dfq.Date == i].index, 'PAY'] = df_peak.loc[i].PAY

    dfq.loc[0, 'PAY'] = dfq[:31]['PAY'].min()

    # Интерполяция временного ряда
    dfi = dfq.copy()
    dfi['PAY'] = dfi['PAY'].interpolate(method='linear', order=5)  # 95% 90%

    # Новый интерполированый ряд с пиками изначального ряда
    dfi = dfi.set_index(pd.DatetimeIndex(dfi.Date))
    dfi = dfi.drop(['Date'], axis=1)

    scaler_pick = MinMaxScaler()
    df_scal_pick = scaler_pick.fit_transform(dfi)
    df_scal_pick = pd.DataFrame(df_scal_pick, columns=dfi.columns)

    start_date = start_date - datetime.timedelta(days=180
                                                 )
    test_pick = df_scal_pick[dfi.index >= start_date]
    train_pick = df_scal_pick[dfi.index < start_date]

    train_x_pick = []
    for i in range(input_len, len(train_pick)):
        train_x_pick.append(train_pick.iloc[i - input_len:i])

    train_y_pick = train_pick.iloc[input_len:]
    train_x_pick = np.array(train_x_pick)
    train_y_pick = np.array(train_y_pick)
    train_x_pick = train_x_pick[:, :, 0]
    train_y_pick = train_y_pick[:, 0]

    test_x_pick = []
    for i in range(input_len, len(test_pick)):
        test_x_pick.append(test_pick.iloc[i - input_len:i])

    test_y_pick = test_pick.iloc[input_len:]
    test_x_pick = np.array(test_x_pick)
    test_y_pick = np.array(test_y_pick)
    test_x_pick = test_x_pick[:, :, 0]
    test_y_pick = test_y_pick[:, 0]

    lsclf_pick = LSSVR(kernel='linear', C=2, gamma=0.001)
    train_ = np.reshape(train_y_pick, train_y_pick.shape[0])

    lsclf_pick.fit(train_x_pick, train_)

    y_pred_lsc_pick = lsclf_pick.predict(test_x_pick)

    temp_pick = df_scal_pick[-pred_len:]
    temp_pick.PAY = y_pred_lsc_pick
    prediction_lsc_pick = pd.DataFrame(scaler_pick.inverse_transform(temp_pick),
                                       columns=df_scal_pick.columns)

    return prediction_lsc_pick, df_peak


def get_answer(num_model, dff, date_1, date_2, pick_check):
    """ Getting forecasting """
    data_frame = dff.copy()
    data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
    data_frame = data_frame.sort_values(by='Date')
    data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
    data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)

    last_real = data_frame.index[-1]
    date_1 = pd.to_datetime(date_1, format='%d.%m.%Y')
    date_2 = pd.to_datetime(date_2, format='%d.%m.%Y')
    start_date = date_1
    end_date = date_2
    pred_len = (last_real - start_date).days + 1
    input_len = 180
    scaler = MinMaxScaler()
    df_scal = scaler.fit_transform(data_frame)
    df_scal = pd.DataFrame(df_scal, columns=data_frame.columns)

    train = df_scal[data_frame.index <= last_real]

    train_x = []
    for i in range(input_len, len(train)):
        train_x.append(train.iloc[i - input_len:i])
    train_y = train.iloc[input_len:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))

    if pred_len > 0:
        pred_len_new = (end_date - last_real).days
    else:
        pred_len_new = (end_date - start_date).days + 1

    if num_model == 1:
        model = svr_model(train_x, train_y)
        if pred_len > 0 and pred_len_new > 0:
            test_x_copy = train_x[-pred_len].copy()

            y_pred_lsc = []
            for i in range(pred_len):
                y_pred_lsc.append(model.predict([test_x_copy]))
                for k in range(1, len(test_x_copy)):
                    test_x_copy[k - 1] = test_x_copy[k]
                test_x_copy[-1] = y_pred_lsc[-1]

            y_pred = []
            for el_pred in y_pred_lsc:
                y_pred.append(el_pred[0])

            temp = df_scal[-pred_len:]
            temp.PAY = y_pred
            prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            # prediction_lssvr["PAY"] = prediction_lssvr["PAY"]

            indexes = pd.DatetimeIndex(data_frame.index[-pred_len:])
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr = prediction_lssvr.set_index(indexes)

            if pick_check == 2:
                prediction_lsc_pick, df_peak = get_picks(data_frame,
                                                         start_date,
                                                         input_len,
                                                         pred_len)
                prediction_lsc_pick = prediction_lsc_pick\
                    .set_index(pd.DatetimeIndex(data_frame.index[-pred_len:]))
                prediction_lsc_copy = prediction_lssvr.copy()
                df_copy = df_peak.index[df_peak.index >= start_date].copy()

                for i in range(len(prediction_lssvr)):
                    for j in range(len(df_copy)):
                        if (prediction_lssvr.index[i:i + 1]) == (df_copy[j:j + 1]):
                            prediction_lsc_copy.PAY[i:i + 1] = prediction_lsc_pick.PAY[i:i + 1]
                prediction_lssvr = prediction_lsc_copy.copy()

            test_pred = train_x[-1].copy()
            for i in range(1, len(test_pred)):
                test_pred[i - 1] = test_pred[i]
            test_pred[len(test_pred) - 1] = train_y[-1]
            y_pred_new = []

            for i in range(pred_len_new):
                y_pred_new.append(model.predict([test_pred]))
                for k in range(1, len(test_pred)):
                    test_pred[k - 1] = test_pred[k]
                test_pred[-1] = y_pred_new[-1]

            y_pred = []
            for el_pred in y_pred_new:
                y_pred.append(el_pred[0])

            temp = df_scal[:pred_len_new]
            temp.PAY = y_pred
            prediction_lssvr_new = pd.DataFrame(scaler.inverse_transform(temp),
                                                columns=df_scal.columns)
            # prediction_lssvr_new["PAY"] = prediction_lssvr_new["PAY"]

            start_date = last_real + datetime.timedelta(days=1)

            res = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            )

            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr_new = prediction_lssvr_new.set_index(indexes)

            prediction_lssvr = prediction_lssvr.append(prediction_lssvr_new)
            # prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr
        if pred_len_new <= 0 < pred_len:
            n_days = (last_real - start_date).days
            n_end = (last_real - end_date).days
            test_x_copy = train_x[-n_days - 1].copy()
            y_pred_lsc = []
            for i in range(pred_len - n_end):
                y_pred_lsc.append(model.predict([test_x_copy]))
                for k in range(1, len(test_x_copy)):
                    test_x_copy[k - 1] = test_x_copy[k]
                test_x_copy[-1] = y_pred_lsc[-1]

            y_pred = []

            for el_pred in y_pred_lsc:
                y_pred.append(el_pred[0])

            if n_end != 0:
                temp = df_scal[-n_days - 1:-n_end]
                temp.PAY = y_pred
                prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp),
                                                columns=df_scal.columns)

                prediction_lssvr["PAY"] = prediction_lssvr["PAY"]

                indexes = pd.DatetimeIndex(data_frame.index[-n_days - 1:-n_end])
                indexes = indexes.strftime('%d.%m.%Y')
                prediction_lssvr = prediction_lssvr.set_index(indexes)
            else:
                temp = df_scal[-n_days - 1:]
                temp.PAY = y_pred
                prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp),
                                                columns=df_scal.columns)

                prediction_lssvr["PAY"] = prediction_lssvr["PAY"]

                indexes = pd.DatetimeIndex(data_frame.index[-n_days - 1:])
                indexes = indexes.strftime('%d.%m.%Y')
                prediction_lssvr = prediction_lssvr.set_index(indexes)

            if pick_check == 2:
                prediction_lsc_pick, df_peak = get_picks(data_frame,
                                                         start_date,
                                                         input_len,
                                                         pred_len)
                prediction_lsc_pick = prediction_lsc_pick\
                    .set_index(pd.DatetimeIndex(data_frame.index[-pred_len:]))
                prediction_lsc_copy = prediction_lssvr.copy()
                df_copy = df_peak.index[df_peak.index >= start_date].copy()

                for i in range(len(prediction_lssvr)):
                    for j in range(len(df_copy)):
                        if (prediction_lssvr.index[i:i + 1]) == (df_copy[j:j + 1]):
                            prediction_lsc_copy.PAY[i:i + 1] = prediction_lsc_pick.PAY[i:i + 1]
                prediction_lssvr = prediction_lsc_copy.copy()
            # prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr
        if (pred_len <= 0 < pred_len_new)  \
                and start_date == (last_real + datetime.timedelta(days=1)):
            test_pred = train_x[-1].copy()
            for i in range(1, len(test_pred)):
                test_pred[i - 1] = test_pred[i]
            test_pred[len(test_pred) - 1] = train_y[-1]
            y_pred_new = []

            for i in range(pred_len_new):
                y_pred_new.append(model.predict([test_pred]))
                for k in range(1, len(test_pred)):
                    test_pred[k - 1] = test_pred[k]
                test_pred[-1] = y_pred_new[-1]

            y_pred = []
            for el_pred in y_pred_new:
                y_pred.append(el_pred[0])

            temp = df_scal[-pred_len_new:]
            temp.PAY = y_pred
            prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
            # prediction_lssvr["PAY"] = prediction_lssvr["PAY"]

            start_date = last_real + datetime.timedelta(days=1)

            res = pd.date_range(
                min(start_date, end_date),
                max(start_date, end_date)
            )

            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            prediction_lssvr = prediction_lssvr.set_index(indexes)
            # prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

            return prediction_lssvr

        n_days = (end_date - last_real).days
        test_pred = train_x[-1].copy()
        for i in range(1, len(test_pred)):
            test_pred[i - 1] = test_pred[i]
        test_pred[len(test_pred) - 1] = train_y[-1]
        y_pred_new = []

        for i in range(n_days):
            y_pred_new.append(model.predict([test_pred]))
            for k in range(1, len(test_pred)):
                test_pred[k - 1] = test_pred[k]
            test_pred[-1] = y_pred_new[-1]

        y_pred = []
        for el_pred in y_pred_new:
            y_pred.append(el_pred[0])

        temp = df_scal[-n_days:]
        temp.PAY = y_pred
        prediction_lssvr = pd.DataFrame(scaler.inverse_transform(temp), columns=df_scal.columns)
        # prediction_lssvr["PAY"] = prediction_lssvr["PAY"]

        start_date = last_real + datetime.timedelta(days=1)

        res = pd.date_range(
            min(start_date, end_date),
            max(start_date, end_date)
        )

        indexes = pd.DatetimeIndex(res)
        indexes = indexes.strftime('%d.%m.%Y')
        prediction_lssvr = prediction_lssvr.set_index(indexes)
        # prediction_lssvr["PAY"] = round(prediction_lssvr["PAY"], 2)

        n_end = (start_date - last_real).days
        return prediction_lssvr[n_end - 1:]
    return None

if __name__ == '__main__':
    df = preprocessing()
    df = df.toPandas()
    data = get_answer(1, df, '01.11.2022', '01.07.2023', 1)
    plt.figure(figsize=(25, 12))
    plt.plot(data.index, data.PAY, label="prediction",
             alpha=.7)
    plt.show()
    print(data)
