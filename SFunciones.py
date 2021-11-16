from alpha_vantage.timeseries import TimeSeries

def get_dataAV(ticker, interval, custom="NA", outformat="pandas"):
	
	""" Info extractor with AV https://github.com/RomelTorres/alpha_vantage
	params:
		    ticker: AMZN, AAPL, etc..
		    interval: daily(d), weekly(w), monthly(m), custom(c)
            custom: custom interval 5min, 15 min, 60 min, etc..
		    outformat: formato salida, 'pandas'
	    return:
		    dataframe with data from ticker in interval 
    """
	
	outputsize = 'compact'
	key = 'AACEU06WTJ7ZFOV'

	ts = TimeSeries(key, output_format=outformat)

	if(interval == 'd'):
		return ts.get_daily_adjusted(ticker)
	elif(interval == 'w'):
		return ts.get_weekly_adjusted(ticker)
	elif(interval == 'm'):
		return ts.get_monthly_adjusted(ticker)
	elif(interval == 'c'):
		return ts.get_intraday(ticker, interval=custom)
	return
		
	