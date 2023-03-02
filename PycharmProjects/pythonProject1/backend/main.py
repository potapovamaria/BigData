"""
Request Controller Module
"""
from pathlib import Path
import json
from flask_cors import CORS, cross_origin  # type: ignore
from flask import Flask, jsonify, request
import pandas as pd  # type: ignore

from payment_analytics.analytics import most_payment_day, \
                                        most_payment_day_of_week, \
                                        decomposition, \
                                        get_price
from prediction.forecasting import get_answer
from backend.preprocessing_data import preprocessing
from backend.common import cache


def make_data():
    """ Making preprocessing data """
    data_frame = preprocessing()
    data_frame = data_frame.toPandas()
    return data_frame


def get_df():
    """ Getting data in necessary format """
    my_value = cache.get("df_new")
    data_frame = my_value
    res = pd.date_range(
        data_frame['PAY_DATE'][0],
        data_frame['PAY_DATE'].to_list()[-1]
    )
    indexes = pd.DatetimeIndex(res)
    indexes = indexes.strftime('%d.%m.%Y')
    data_frame = data_frame.set_index(indexes)
    data_frame = data_frame.drop(['PAY_DATE'], axis=1)
    data_frame = data_frame.to_dict()
    data_frame = jsonify(data_frame)
    return data_frame

def create_application() -> Flask:
    """ Request Controller """
    app_new = Flask(__name__)
    cache.init_app(app=app_new, config={"CACHE_TYPE": "filesystem", 'CACHE_DIR': Path('/tmp')})
    cache.set("flag", False)
    CORS(app_new)
    app_new.config['CORS_HEADERS'] = 'Content-Type'
    app_new.config['JSON_SORT_KEYS'] = False

    @app_new.route("/getdata", methods=["POST"])
    @cross_origin()
    def get_data():
        if request.method == 'POST':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")
            data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
            data_frame = data_frame.sort_values(by='Date')
            data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
            data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)
            start_date = request.json.get('start_date')
            end_date = request.json.get('end_date')
            data = data_frame[data_frame.index >= start_date].copy()
            data = data[data.index <= end_date]
            if len(data) == 0:
                return "No data"

            res = pd.date_range(
                data.index[0],
                data.index.to_list()[-1]
            )
            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            data = data.set_index(indexes)
            data = data.to_dict()
            data = jsonify(data)
            return data
        return "No data"

    @app_new.route("/getWeekday", methods=["GET"])
    @cross_origin()
    def get_weekday():
        if request.method == 'GET':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")
            week_day = most_payment_day_of_week(data_frame)
            return json.dumps(week_day)
        return "No data"

    @app_new.route("/getDay", methods=["GET"])
    @cross_origin()
    def get_day():
        if request.method == 'GET':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")
            day = most_payment_day(data_frame)
            return json.dumps(day)
        return "No data"

    @app_new.route("/getDecomposition", methods=["GET"])
    @cross_origin()
    def get_decomposition():
        if request.method == 'GET':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")

            trend, seasonal, resid = decomposition(data_frame)
            data = {'trend': trend, 'seasonal': seasonal, 'resid': resid}
            json_data = json.dumps(data)
            return json_data
        return "No data"

    @app_new.route("/getPrice", methods=["GET"])
    @cross_origin()
    def get_new_price():
        if request.method == 'GET':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")
            data = get_price(data_frame)
            data.index = data.index.strftime('%d.%m.%Y')
            data = data.to_dict()
            data = jsonify(data)
            return data
        return "No data"

    @app_new.route("/prediction/payment", methods=["POST"])
    @cross_origin()
    def payment_prediction():
        if request.method == 'POST':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            data_frame = cache.get("df_new")
            start_date = request.json.get('start_date')
            end_date = request.json.get('end_date')
            model_num = request.json.get("model_num")
            model_num = int(model_num)
            pick_check = request.json.get('enter_pick')
            pick_check = int(pick_check)
            y_pred = get_answer(model_num, data_frame, start_date, end_date, pick_check)
            y_pred = y_pred.to_dict()
            y_pred = jsonify(y_pred)
            return y_pred
        return "No data"

    @app_new.route("/getdatapred", methods=["POST"])
    @cross_origin()
    def get_orig_data():
        if request.method == 'POST':
            if cache.get("flag") is False:
                data_frame = make_data()
                cache.set("df_new", data_frame, timeout=0)
                cache.set("flag", True)
            start_date = request.json.get('start_date')
            end_date = request.json.get('end_date')
            data_frame = cache.get("df_new")
            data_frame['Date'] = pd.to_datetime(data_frame['PAY_DATE'], format='%Y-%m-%d')
            data_frame = data_frame.sort_values(by='Date')
            data_frame = data_frame.set_index(pd.DatetimeIndex(data_frame['Date']))
            data_frame = data_frame.drop(['Date', 'PAY_DATE'], axis=1)

            data = data_frame[data_frame.index >= start_date].copy()
            data = data[data.index <= end_date]
            if len(data) == 0:
                return "No data"

            res = pd.date_range(
                data.index[0],
                data.index.to_list()[-1]
            )
            indexes = pd.DatetimeIndex(res)
            indexes = indexes.strftime('%d.%m.%Y')
            data = data.set_index(indexes)
            data = data.to_dict()
            data = jsonify(data)
            return data
        return "No data"

    return app_new


if __name__ == "__main__":
    app = create_application()
    app.run(host="localhost", port=5000, debug=True)
