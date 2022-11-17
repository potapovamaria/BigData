# import pandas as pd
# from pyspark.ml.feature import MinMaxScaler
#
# import preprocessing_data
#
#
# def get_answer():
#     df = preprocessing_data.preprocessing()
#     scaler = MinMaxScaler()
#     df_scal = scaler.fit_transform(df)
#     df_scal = pd.DataFrame(df_scal, columns=df.columns)
#     LAST_REAL = df.PAY_DATE[-1]
#     train = df_scal[df.PAY_DATE <= LAST_REAL]
#
#     train_x = []