import sqlite3
import numpy as np
from matplotlib import pyplot as plt


def get_plots():
	pass

def price_plot(db_path, title, best_price=1.0):
	query = "SELECT INSERT_TIME, (BEST_PRICE/{}) FROM STATS WHERE REGION_SOLUTION='us-east-1'".format(best_price)
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