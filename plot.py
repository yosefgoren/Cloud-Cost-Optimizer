import sqlite3
import numpy as np
from matplotlib import pyplot as plt

def price_plot(db_path, title, best_price=1.0):
	query = f"SELECT INSERT_TIME, (BEST_PRICE/{best_price}) FROM STATS"
	# query = "SELECT INSERT_TIME, (BEST_PRICE/{}) FROM STATS WHERE REGION_SOLUTION='us-east-1'".format(best_price)
	#title = "price plot, 14 comp"
	x_label = "time (seconds)"
	y_label = "price"
	plot(db_path, query, title, x_label, y_label)

def nodes_count_plot(db_path):
	query = "SELECT INSERT_TIME, NODES_COUNT FROM STATS WHERE REGION_SOLUTION='ap-south-1'"
	title = "nodes count plot, 14 comp"
	x_label = "time (seconds)"
	y_label = "number of nodes"
	plot(db_path, query, title, x_label, y_label)

def plot(db_path, query, title, x_label, y_label):
	conn = sqlite3.connect(db_path)

	res = conn.execute(query)

	arr = np.array(list(res))
	arr[:,0] -= np.min(arr[:,0])
	plt.plot(arr[:,0], arr[:,1])
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

	conn.close()

def plot_single_time_line(db_path: str, title: str, x_label: str = "x", y_label: str = "y"):
	query = "SELECT INSERT_TIME, (BEST_PRICE) FROM STATS ORDER BY INSERT_TIME; "
	conn = sqlite3.connect(db_path)

	res = conn.execute(query)

	arr = np.array(list(res))
	xs = [arr[0,0]]
	ys = [arr[0,1]]
	for i in range(1, arr.shape[0]):
		if arr[i, 1] <= ys[-1]:
			xs.append(arr[i, 0])
			ys.append(arr[i, 1])

	# plt.plot(arr[:,0], arr[:,1])
	plt.plot(xs, ys)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

	conn.close()


plot_single_time_line("./experiments/overnight_bias_candidates/stats/0.sqlite3", "a")