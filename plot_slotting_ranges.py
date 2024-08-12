import os
import csv
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator, MultipleLocator

def read_data(directory):

	demand_data = {}
	file_pattern = re.compile(r'demand_D[a-f]_overview\.')
	#file_pattern = re.compile(r'demand_D(a|f)_overview\.')

	for filename in os.listdir(directory):
		if file_pattern.match(filename):
			file_path = os.path.join(directory, filename)
			with open(file_path, 'r') as file:
				reader = csv.reader(file)
				lines = list(reader)
				if len(lines) > 1:
					demand_name = lines[1][0]
					parts = lines[1][1:]
					last_three_parts = parts[-3:]
					demand_data[demand_name] = last_three_parts

	return demand_data

def config_weigth_data():

	ranges = []

	range_w2 = {
		'Da': [0.052, 10],
		'Db': [0.059, 1.295],
		'Dc': [0.058, 6.087],
		'Dd': [0.05, 0.79],
		'De': [0.058, 1],
		'Df': [0.1, 0.38],
		'Dr': [0.01, 1.6],
		'Dp': [0.01, 1]
	}
	range_w3 = {
		'Da': [0.052, 1.039],
		'Db': [0.059, 0.794],
		'Dc': [0.058, 0.7],
		'Dd': [0.05, 0.54],
		'De': [0.058, 0.058],
		'Df': [0.068, 0.068],
		'Dr': [1, 1],
		'Dp': [1, 1]
	}

	demands = list(range_w2.keys())

	for demand in demands:
		w1 = [1, 1]  # w1 values are always [1, 1]
		w2 = range_w2[demand]
		w3 = range_w3[demand]
		ranges.append((w1, w2, w3))

	return ranges

def config_weigth_gall_data():

	ranges = []

	range_w2 = {
		'Dr': [0.01, 1.6],
		'Dp': [0.01, 1]
	}
	range_w3 = {
		'Dr': [1, 1],
		'Dp': [1, 1]
	}

	demands = list(range_w2.keys())

	for demand in demands:
		w1 = [1, 1]  # w1 values are always [1, 1]
		w2 = range_w2[demand]
		w3 = range_w3[demand]
		ranges.append((w1, w2, w3))

	return ranges

def plot_data(demand_data):
	"""
	Plots the extracted data.

	Args:
	- demand_data (dict): A dictionary with demand names as keys and the last three parts of the second line as values.
	"""
	demands = list(demand_data.keys())
	pods_per_item = []
	items_per_pod = []
	pods_per_sku = []


	for demand in demands:
		pods_per_item.append(eval(demand_data[demand][0]))
		items_per_pod.append(eval(demand_data[demand][1]))
		pods_per_sku.append(eval(demand_data[demand][2]))

	weight_ranges = config_weigth_data()

	# Colors for different demands
	colors = ['r', 'g', 'b', 'c', 'm', 'orange']

	def create_bar_plot(data, ylabel, title):
		fig, ax = plt.subplots(figsize=(12, 8))
		tick_labels = []

		bar_label = ['Demand A', 'Demand B', 'Demand C', 'Demand D', 'Demand E', 'Demand F']
		for i, demand in enumerate(demands):
			color = colors[i % len(colors)]
			for j, label in enumerate(['t', "t'", "t''"]):
				value = data[i][j]
				min_val, max_val = min(value), max(value)
				height = max_val - min_val if max_val != min_val else 0.005

				alpha = 1.0 if j == 0 else 0.5 if j == 1 else 0.2
				tick_labels.append(chr(65 + j))  # Append 'A', 'B', 'C'
				#bar_label = f'{demand}_{label}'.replace('_t', '').replace("_t'", '').replace("_t''", '')
				ax.bar(f'{demand}_{label}', height, bottom=min_val, color=color, alpha=alpha, edgecolor='none',
				       label=bar_label[i] if j == 0 else "")
				ax.bar(f'{demand}_{label}', height, bottom=min_val, color='none', edgecolor=color, linewidth=1.2,
				       alpha=1.0)

		ax.set_xlabel('Classes per demand profile')
		ax.set_ylabel(ylabel)
		ax.set_title(title)

		if ylabel == 'Weight':
			# Function x**(1/2)
			def forward(x):
				return x ** (1 / 2)

			def inverse(x):
				return x ** 2
			ax.set_yscale('function', functions=(forward, inverse))

			coordinates = np.concatenate([np.arange(0, 1, 0.25), np.arange(1, 2, 0.5), np.arange(2, 10, 2)])
			ax.yaxis.set_major_locator(FixedLocator(coordinates))

		# Add grid
		ax.grid(True, which='both', linestyle='--', linewidth=0.5)

		# Adjust x-ticks
		ax.set_xticklabels(tick_labels, rotation=45, ha='right')

		# Place legend outside the plot
		ax.legend(loc='lower left', bbox_to_anchor=(0.68, 0.5))
		plt.savefig(os.path.join(directory, f'{ylabel.replace("/", "_")}_rangeplot.png'))
		plt.show()
		plt.close(fig)

	#create_bar_plot(pods_per_item, 'Pods/Item', 'Range of pods/item for classes per demand profile')
	#create_bar_plot(items_per_pod, 'Items/Pod', 'Range of items/pod classes per demand profile')
	#create_bar_plot(pods_per_sku, 'Pods/SKU', 'Range of pods/SKU classes per demand profile')
	create_bar_plot(weight_ranges, ylabel='Weight', title='Range of weights classes per demand profile')

def plot_data_gall(demand_data):
	"""
	Plots the extracted data.

	Args:
	- demand_data (dict): A dictionary with demand names as keys and the last three parts of the second line as values.
	"""
	demands = list(demand_data.keys())
	pods_per_item = []
	items_per_pod = []
	pods_per_sku = []


	for demand in demands:
		pods_per_item.append(eval(demand_data[demand][0]))
		items_per_pod.append(eval(demand_data[demand][1]))
		pods_per_sku.append(eval(demand_data[demand][2]))

	weight_ranges = config_weigth_gall_data()

	# Colors for different demands
	colors = ['r', 'orange', 'yellow']

	def create_bar_plot(data, ylabel, title):
		fig, ax = plt.subplots(figsize=(12, 8))
		tick_labels = []

		for i, demand in enumerate(demands):
			color = colors[i % len(colors)]
			for j, label in enumerate(['t', "t'", "t''"]):
				value = data[i][j]
				min_val, max_val = min(value), max(value)
				height = max_val - min_val if max_val != min_val else 0.005
				alpha = 1.0 if j == 0 else 0.5 if j == 1 else 0.2
				tick_labels.append(chr(65 + j))  # Append 'A', 'B', 'C'
				bar_label = f'{demand}_{label}'.replace('_t', '').replace("_t'", '').replace("_t''", '')
				ax.bar(f'{demand}_{label}', height, bottom=min_val, color=color, alpha=alpha, edgecolor='none',
				       label=bar_label if j == 0 else "")
				ax.bar(f'{demand}_{label}', height, bottom=min_val, color='none', edgecolor=color, linewidth=1.2,
				       alpha=1.0)

		ax.set_xlabel('Classes per demand profile')
		ax.set_ylabel(ylabel)
		ax.set_title(title)

		if ylabel == 'Weight':
			# Function x**(1/2)
			def forward(x):
				return x ** (1 / 2)

			def inverse(x):
				return x ** 2
			ax.set_yscale('function', functions=(forward, inverse))

			coordinates = np.concatenate([np.arange(0, 1, 0.25), np.arange(1, 2, 0.5), np.arange(2, 10, 2)])
			ax.yaxis.set_major_locator(FixedLocator(coordinates))

		# Add grid
		ax.grid(True, which='both', linestyle='--', linewidth=0.5)

		# Adjust x-ticks
		ax.set_xticklabels(tick_labels, rotation=45, ha='right')

		# Place legend outside the plot
		ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
		plt.savefig(os.path.join(directory, f'{ylabel.replace("/", "_")}_rangeplot.png'))
		plt.show()
		plt.close(fig)



	create_bar_plot(pods_per_item, 'Pods/Item', 'Range of pods/item for classes per demand profile')
	create_bar_plot(items_per_pod, 'Items/Pod', 'Range of items/pod classes per demand profile')
	create_bar_plot(pods_per_sku, 'Pods/SKU', 'Range of pods/SKU classes per demand profile')
	create_bar_plot(weight_ranges, ylabel='Weight', title='Range of weights classes per demand profile')


# Directory containing the files

plt.rcParams.update({'font.size': 20,  # controls default text sizes
		                     'axes.titlesize': 20,  # fontsize of the axes title
		                     'axes.labelsize': 20,  # fontsize of the x and y labels
		                     'xtick.labelsize': 20,  # fontsize of the tick labels
		                     'ytick.labelsize': 20,  # fontsize of the tick labels
		                     'legend.fontsize': 20,  # fontsize of the legend
		                     'figure.titlesize': 20  # fontsize of the figure title
		                     })

directory = r'C:\Users\pnl0j327\PycharmProjects\pythonProject1\specific_weights_steps_S100_d1000_T120'
#directory = r'C:\Users\pnl0j327\PycharmProjects\pythonProject1\Gall_step12_S100_d1000_T120'

demand_data = read_data(directory)

plot_data(demand_data)
#plot_data_gall(demand_data)



