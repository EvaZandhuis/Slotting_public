import pandas as pd
import csv
import os
import matplotlib.pyplot as plt



def read_gall_demand(filename, filepath):
	D_regular = []
	D_dec = []

	if filepath is None:
		path = filename
	else:
		path = os.path.join(filepath, filename)
	with open(path, mode='r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			date = float(row['afleverdatum'])
			SKU_i = float(row['SKU_nr'])
			demand_value = float(row['geleverd'])

			#if date between jfjf
			#D_regular.append(demand_value)
		#if date is december:
			#D_dec.append(demand_value)

	return D_regular, D_dec


def filter_data_by_date(file_path, filter_date_begin, filter_date_end):

	df = pd.read_csv(file_path, delimiter=';')
	df.columns = df.columns.str.strip()
	df['afleverdatum'] = pd.to_datetime(df['afleverdatum'], dayfirst=True)
	filtered_df = df[(df['afleverdatum'] >= filter_date_begin) & (df['afleverdatum'] <= filter_date_end)]
	result_df = filtered_df.groupby('artikelnummer', as_index=False)['geleverd'].sum()

	return result_df


def plot_data(data):
	# Sort the data by 'geleverd' in descending order
	sorted_data = data.sort_values(by='geleverd', ascending=False)

	# Generate x-axis labels from 1 to the number of values
	x_labels = range(1, len(sorted_data) + 1)

	# Extract the 'geleverd' values as y-axis values
	y_values = sorted_data['geleverd'].values

	# Plotting
	plt.figure(figsize=(10, 5))
	plt.bar(x_labels, y_values, color='blue')

	# Add semi-transparent bars
	total_skus = len(sorted_data)  # Total number of SKUs (or artikelnummers)

	top_labels = ['20%', '30%', '50%']
	right_labels = ['75%', '20%', '5%']

	widths_input = [20, 50]
	widths = [widths_input[0]*total_skus/100, (widths_input[1] - widths_input[0])*total_skus /100, (100-widths_input[1])*total_skus/100]
	sku_part = [int((total_skus * widths_input[0]) /100), int((total_skus * widths_input[1])/100)]
	print(100*sum(sorted_data['geleverd'].values[0:sku_part[0]])/sum(sorted_data['geleverd'].values))
	print(100*sum(sorted_data['geleverd'].values[sku_part[0]:sku_part[1]])/sum(sorted_data['geleverd'].values))
	print(100*sum(sorted_data['geleverd'].values[sku_part[1]:])/sum(sorted_data['geleverd'].values))
	heights = [sum(sorted_data['geleverd'].values[0:sku_part[0]]) / len(filtered_data['geleverd'].values[0:sku_part[0]]) ,sum(filtered_data['geleverd'].values[sku_part[0]:sku_part[1]]) / len(filtered_data['geleverd'].values[sku_part[0]:sku_part[1]]), sum(filtered_data['geleverd'].values[sku_part[1]:]) / len(filtered_data['geleverd'].values[sku_part[1]:])]
	#heights = [sum(filtered_data['geleverd'].values[0:sku_part[0]])/widths[0], sum(filtered_data['geleverd'].values[sku_part[0]:sku_part[1]])/widths[1], sum(filtered_data['geleverd'].values[sku_part[1]:])/widths[2]]
	#heights = [suwidths_input[0] * sum_geleverd / sum_geleverd, 0.3 * sum_geleverd / sum_geleverd, 0.2 * sum_geleverd / sum_geleverd]
	print(heights[0], heights[1], heights[2])
	current_width = 0
	colors = ['red', 'orange', 'yellow']
	for i in range(len(widths)):
		plt.bar(current_width + 1, heights[i], color=colors[i], alpha=0.5, width=widths[i], align='edge')
		plt.text(current_width + widths[i] / 2, heights[i] + 5, top_labels[i], ha='center', va='center')
		plt.text(current_width + widths[i], heights[i] / 2, right_labels[i], ha='right', va='center', rotation=270)
		current_width += widths[i]



	# Labeling the axes
	plt.xlabel('SKUs [i]')
	plt.ylabel('Demand [items]')

	# Title of the plot
	plt.title('ABC class configuration for Gall&Gall regular demand')

	# Set y-axis limit
	plt.ylim(0, 250)

	# Show the plot
	plt.savefig("plot_gall_regular_demand.png", bbox_inches='tight')
	plt.show()
	plt.close()


file_path = 'Data_case_study_items_per_sku.csv'
filter_date_begin = pd.to_datetime('2024-05-01')
filter_date_end = pd.to_datetime('2024-06-01')

filtered_data = filter_data_by_date(file_path, filter_date_begin, filter_date_end)
plot_data(filtered_data)
