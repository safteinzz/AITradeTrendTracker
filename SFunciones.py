from alpha_vantage.timeseries import TimeSeries #https://github.com/RomelTorres/alpha_vantage
from sklearn import preprocessing
import pandas as pd
import numpy as np

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def get_dataAV(ticker, interval, custom = "NA", outformat = "pandas", outputsize = "full", key = "AACEU06WTJ7ZFOV"):
	
	""" Info extractor with AV
	params:
		    ticker: AMZN, AAPL, etc..
		    interval: daily(d), weekly(w), monthly(m), custom(c)
            custom: custom interval 5min, 15 min, 60 min, etc..
		    outformat: formato salida, 'pandas'
            outputsize: compact (100 lines), full (full history)
            key: api key
	    return:
		    @outformat (default pandas df) with data from ticker in interval 
    """
	
	ts = TimeSeries(key, output_format=outformat)

	if(interval == 'd'):
		return ts.get_daily_adjusted(ticker, outputsize=outputsize)[0]
	elif(interval == 'w'):
		return ts.get_weekly_adjusted(ticker)[0]
	elif(interval == 'm'):
		return ts.get_monthly_adjusted(ticker)[0]
	elif(interval == 'c'):
		return ts.get_intraday(ticker, interval=custom, outputsize=outputsize)[0]
	return
		
def prepare_data(df):


    # Scale values 0 to 1
    column_scaler = {}
    for columnName in df.columns:
        # print(columnName)
        scaler = preprocessing.MinMaxScaler()
        df[columnName] = scaler.fit_transform(np.expand_dims(df[columnName].values, axis=1))
        column_scaler[columnName] = scaler

    # Hay que agregar datos como noticias, deberia de sacar positividad de las noticias medias en referencia al tema para agregarlo como positividad de las noticias como linea adicional

    

    # Add date as column, it is index by default
    # if "date" not in df.columns:
    #     print('drogas')
    #     df["date"] = df.index




    # Mezclar datos para mejor confidence


    return df
