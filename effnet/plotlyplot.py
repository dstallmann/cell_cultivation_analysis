import argparse
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from outliers import smirnov_grubbs as grubbs

store_plots = True

np.random.seed(1)

def createPlot(path, directory, title, warm_start):
	dataframe = pd.read_csv(directory + "/logs/" + path, header=None, index_col=False)
	dataframe.columns = ['y']
	dataframe.drop([0, warm_start])
	fig = px.line(dataframe, y="y", title=title)
	if store_plots:
		plotly.offline.plot(fig, filename=directory + "/" + title + ".html") # Includes fig.show()
	else:
		fig.show()

def createPlotWithMean(path, directory, title, warm_start, mean):
	dataframe = pd.read_csv(directory + "/logs/" + path, header=None, index_col=False)
	dataframe.columns = ['y']
	dataframe.drop([0, warm_start])
	fig = px.line(dataframe, y="y", title=title)
	spline_data = np.mean(dataframe.to_numpy().flatten().reshape(-1, mean), axis=1)
	fig.add_trace(go.Scatter(x=list(range(0, len(dataframe))[::mean]), y=spline_data, name="Mean (" + str(mean) + ")", line_shape='spline', mode='lines+markers'))
	if store_plots:
		plotly.offline.plot(fig, filename=directory + "/" + title + ".html")  # Includes fig.show()
	else:
		fig.show()

def createPlots(warm_start, mean_size, directory):
	#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#parser.add_argument('--dir', help='directory', default="results/resultDefault", required=False)
	#args = parser.parse_args()
	#if directory == "":
		#directory = "results/" + args.dir

	createPlot("Decoder_syn_log.txt", directory, "Decoder synth loss", warm_start)
	createPlot("Decoder_nat_log.txt", directory, "Decoder natural loss", warm_start)
	createPlot("KLD_syn_log.txt", directory, "KLD synth loss", warm_start)
	createPlot("KLD_nat_log.txt", directory, "KLD natural loss", warm_start)
	createPlot("Regressor_syn_log.txt", directory, "Regressor synth loss", warm_start)
	createPlot("Regressor_nat_log.txt", directory, "Regressor natural loss", warm_start)

	createPlotWithMean("Correct_nat_log.txt", directory, "Correct natural", warm_start, mean_size)
	createPlotWithMean("Correct_syn_log.txt", directory, "Correct synth", warm_start, mean_size)

def comparisonPlot(logname, dirs, warm_start, suffix="", mean_size=0):
	fig = go.Figure()
	xstring = "Epoch" + suffix
	ystring = "Value"
	fig.update_layout(
		title=go.layout.Title(text=logname, xref="paper", x=0),
		xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=xstring, )),
		yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=ystring, )),
		font=dict(
			family="Arial",
			size=16,
			color="#505050"
		)
	)

	for i in range(0, len(dirs)):
		dataframe = pd.read_csv("../logs/" + dirs[i] + "/" + logname + "_log.txt", header=None, index_col=False)
		data = dataframe.to_numpy().flatten()
		data = data[warm_start:] # Ignore first warm_start epochs
		#data = grubbs.test(data, alpha=1.0) # 0.999) # Removes outliers
		if mean_size != 0:
			remainder = len(data) % mean_size
			while remainder != 0 and remainder != mean_size:
				data = np.append(data, data[-1])
				remainder += 1

			spline_data = np.mean(data.reshape(-1, mean_size), axis=1)
			fig.add_trace(go.Scatter(x=list(range(0, len(data))[::mean_size]), y=spline_data, name=dirs[i], line_shape='linear')) # line_shape='spline'
		else:
			datasum = 0
			for d in data:
				datasum += d
			avg = datasum/len(data)
			for idx in range(len(data)):
				if data[idx] > avg*100:
					data[idx] = (data[idx-1]+data[idx+1])/2 # Avg
			fig.add_trace(go.Scatter(x=list(range(0, len(data))), y=data, mode="lines", name=dirs[i]))

	plotly.offline.plot(fig, filename="../logs/comp_" + logname + ".html")  # includes fig.show()

if __name__ == '__main__':
	# createPlots(100, 500)

	result_dirs = []
	for name in os.listdir("../logs/"):
		result_dirs.append(name)
	print(result_dirs)
	# result_dirs = ["dir1", "dir2", etc]
	prefix = ""
	if prefix != "":
		for x in range(len(result_dirs)):
			result_dirs[x] = prefix + result_dirs[x]
	comparisonPlot("Train_MAE", result_dirs, 0, "")
	comparisonPlot("Test_MAE", result_dirs, 0, " x10")
	comparisonPlot("Train_DEV", result_dirs, 0, "")
	comparisonPlot("Test_DEV", result_dirs, 0, " x10")
	comparisonPlot("Train_COR", result_dirs, 0, "")
	comparisonPlot("Test_COR", result_dirs, 0, " x10")
	
	#comparisonPlot("test_syn_cor", result_dirs, folder, 0, 1)
	#comparisonPlot("test_nat_cor", result_dirs, folder, 0, 1)
	
	#comparisonPlot("test_syn_mae", result_dirs, folder, 0, 1)
	#comparisonPlot("test_nat_mae", result_dirs, folder, 0, 1)
	
	#comparisonPlot("test_syn_dev", result_dirs, folder, 0, 1)
	#comparisonPlot("test_nat_dev", result_dirs, folder, 0, 1)
