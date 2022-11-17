from pathlib import Path
import json
import pandas as pd
import preprocessing_data
import analytics
import os
from flask_cors import CORS, cross_origin
from flask import Flask, Response, jsonify, request, redirect, url_for, flash
import pandas as pd
from common import cache

def make_data():
    df = preprocessing_data.preprocessing()
    df = df.toPandas()
    return df

def get_df():
    my_value = cache.get("df_new")
    df = my_value
    res = pd.date_range(
        df['PAY_DATE'][0],
        df['PAY_DATE'].to_list()[-1]
    )
    indexes = pd.DatetimeIndex(res)
    indexes = indexes.strftime('%d.%m.%Y')
    df = df.set_index(indexes)
    df = df.drop(['PAY_DATE'], axis=1)
    df = df.to_dict()
    df = jsonify(df)
    return df

def create_application() -> Flask:
    app = Flask(__name__)
    cache.init_app(app=app, config={"CACHE_TYPE": "filesystem", 'CACHE_DIR': Path('/tmp')})
    cache.set("flag", False)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['JSON_SORT_KEYS'] = False
    @app.route("/getdata", methods=["GET"])
    @cross_origin()
    def get_data():
        if request.method == 'GET':
            if cache.get("flag") == False:
                df = make_data()
                cache.set("df_new", df)
                cache.set("flag", True)
            return get_df()

    @app.route("/getWeekday", methods=["GET"])
    @cross_origin()
    def get_weekday():
        if request.method == 'GET':
            if cache.get("flag") == False:
                df = make_data()
                cache.set("df_new", df)
                cache.set("flag", True)
            df = cache.get("df_new")
            df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
            df = df.sort_values(by='Date')
            df = df.set_index(pd.DatetimeIndex(df['Date']))
            df = df.drop(['Date', 'PAY_DATE'], axis=1)
            weekDay = analytics.most_payment_day_of_week(df)
            return json.dumps(weekDay.tolist())

    @app.route("/getDay", methods=["GET"])
    @cross_origin()
    def get_day():
        if request.method == 'GET':
            if cache.get("flag") == False:
                df = make_data()
                cache.set("df_new", df)
                cache.set("flag", True)
            df = cache.get("df_new")
            df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
            df = df.sort_values(by='Date')
            df = df.set_index(pd.DatetimeIndex(df['Date']))
            df = df.drop(['Date', 'PAY_DATE'], axis=1)
            day = analytics.most_payment_day(df)
            return json.dumps(day.tolist())
    return app

if __name__== "__main__":
    app = create_application()
    app.run(host="localhost", port=5000, debug=True)
# if __name__ == '__main__':
#     df = preprocessing_data.preprocessing()
    # df = df.toPandas()
    # df['Date'] = pd.to_datetime(df['PAY_DATE'], format='%Y-%m-%d')
    # df = df.sort_values(by='Date')
    # df = df.set_index(pd.DatetimeIndex(df['Date']))
    # # df.index = df.index.dt.strftime('%d.%m.%y')
    # df = df.drop(['Date', 'PAY_DATE'], axis=1)
    # analytics.visualisation(df)
    # analytics.most_payment_day_of_week(df)
    # analytics.most_payment_day(df)
    # analytics.decomposition(df)
    # analytics.check_stationarity(df)

