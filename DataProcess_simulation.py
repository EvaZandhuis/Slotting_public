import csv
import os
import statistics
from collections import defaultdict
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
import ternary
from matplotlib.colors import Normalize

def read_footprints(prefix, path):
    base_path = path
    columns = ['W2', 'W3', 'Za', 'Zb', 'Zc', 'ItemPileOneAvg', 'DistanceTraveled', 'OrderTurnoverTimeAvg', 'OrdersHandled']
    input_file_path = os.path.join(base_path, f"{prefix}", "footprints.csv")
    """
    Reads the specified CSV file, filters rows by the given prefix, extracts the specified columns,
    and writes the result to a new CSV file.
    """
    filtered_data = []

    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            filtered_row = {col: row[col] for col in columns}
            filtered_data.append(filtered_row)

    output_dir = r'C:\Users\pnl0j327\RAWSim-O\RAWSim-O-main\Material\Instances'
    output_file_path = os.path.join(base_path, f"{prefix}", f"filtered_footprints_{prefix}.csv")
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(filtered_data)

def write_avg_statistics_data(prefix):
    base_path = r'C:\Users\pnl0j327\RAWSim-O\RAWSim-O-main\Material\Instances'
    """
    Reads the filtered footprints CSV file, computes averages, minimum, maximum, standard deviation,
    and performs one-sample t-tests for specified columns based on the same values of W2 and W3.
    Writes the results to a new CSV file.
    """
    input_file_path = os.path.join(base_path, f"filtered_footprints_{prefix}.csv")
    output_file_path = os.path.join(base_path, f"filtered_footprints_statistics_weight_{prefix}.csv")

    # Columns to calculate statistics and perform t-tests
    columns = ['ItemPileOneAvg', 'DistanceTraveled', 'OrderTurnoverTimeAvg', 'OrdersHandled']

    # Dictionary to store statistics and t-test results for each (W2, W3) combination
    results = []

    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Group data by (W2, W3) and calculate statistics for each column
        grouped_data = defaultdict(lambda: {col: [] for col in columns})
        for row in reader:
            key = (row['W2'], row['W3'])
            for col in columns:
                grouped_data[key][col].append(float(row[col]))

        overall_means = {col: statistics.mean([value for key, data_dict in grouped_data.items() for value in data_dict[col]]) for col in columns}

        # Calculate means and perform one-sample t-tests for each (W2, W3) combination
        for key, data_dict in grouped_data.items():
            t_test_results = {}
            for col in columns:
                data = data_dict[col]
                mean = statistics.mean(data)
                # Perform one-sample t-test
                t_statistic, p_value = ttest_1samp(data, overall_means[col])
                t_test_results[f'{col}_mean'] = mean
                t_test_results[f'{col}_t_statistic'] = t_statistic
                t_test_results[f'{col}_p_value'] = p_value

            results.append({'W2': key[0], 'W3': key[1], **t_test_results})

    # Write results to a new CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['W2', 'W3']
        for col in columns:
            fieldnames.extend([f'{col}_mean', f'{col}_t_statistic', f'{col}_p_value'])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def write_avg_data(prefix, path):
    base_path = path
    """
    Reads the filtered footprints CSV file, computes averages, minimum, maximum, standard deviation,
    and performs one-sample t-tests for specified columns based on the same values of W2 and W3.
    Writes the results to a new CSV file.
    """
    input_file_path = os.path.join(base_path, f"{prefix}", f"filtered_footprints_{prefix}.csv")
    output_file_path = os.path.join(base_path, f"{prefix}", f"filtered_footprints_statistics_weight_{prefix}.csv")

    # Columns to calculate statistics
    columns = ['ItemPileOneAvg', 'DistanceTraveled', 'OrderTurnoverTimeAvg', 'OrdersHandled']

    # Dictionary to store statistics and count for each (W2, W3) combination
    results = []

    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Group data by (W2, W3) and calculate statistics for each column
        grouped_data = defaultdict(lambda: {col: [] for col in columns})
        for row in reader:
	        key = (round(float(row['W2']), 3), round(float(row['W3']), 3))
	        for col in columns:
		        grouped_data[key][col].append(float(row[col]))

        # Calculate overall means, standard deviations, and count
        overall_means = {col: round(statistics.mean([value for data_dict in grouped_data.values() for value in data_dict[col]]), 2) for col in columns}
        overall_stddevs = {col: round(statistics.stdev([value for data_dict in grouped_data.values() for value in data_dict[col]]), 2) for col in columns}
        overall_count = sum(len(data_dict[columns[0]]) for data_dict in grouped_data.values())

        # Calculate means, standard deviations, and count for each (W2, W3) combination
        for key, data_dict in grouped_data.items():
            result = {'W2': key[0], 'W3': key[1]}
            for col in columns:
                data = data_dict[col]
                mean = round(statistics.mean(data), 2)
                std_dev = round(statistics.stdev(data), 2)
                result[f'{col}_mean'] = mean
                result[f'{col}_stddev'] = std_dev
            result['count'] = len(data_dict[columns[0]])
            results.append(result)

    # Write results to a new CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        # Create the first line of the header
        header1 = ['W2', 'W3']
        for col in columns:
            header1.extend([col, ''])
        #header1.append('Count')

        # Create the second line of the header
        header2 = ['', '']
        for _ in columns:
            header2.extend(['Mean', 'SD'])

        header3 = ['']

        writer = csv.writer(csvfile)
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow([''])


        # Write overall means, standard deviations, and count
        overall_row = ['Total', '']
        for col in columns:
            overall_row.extend([overall_means[col], overall_stddevs[col]])
        overall_row.append(overall_count)
        writer.writerow(overall_row)

        writer.writerow(header3)

        # Write individual results
        for result in results:
            row = [result['W2'], result['W3']]
            for col in columns:
                row.extend([result[f'{col}_mean'], result[f'{col}_stddev']])
            #row.append(result['count'])
            writer.writerow(row)
        writer.writerow([''])

def create_ternary_plot(prefix, path):
    if prefix == 'Da':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 58
        Di_b = 10
        Di_c = 4
    if prefix == 'Db':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 34
        Di_b = 5
        Di_c = 3
    if prefix == 'Dc':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 69
        Di_b = 7
        Di_c = 3
    if prefix == 'Dd':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 40
        Di_b = 3
        Di_c = 2
    if prefix == 'De':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 86
        Di_b = 6
        Di_c = 1
    if prefix == 'Df':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 44
        Di_b = 2
        Di_c = 1
    if prefix == 'Dr':
        Q_a = 20
        Q_b = 30
        Q_c = 50
        Di_a = 37
        Di_b = 7
        Di_c = 1
    if prefix == 'Dp':
        Q_a = 20
        Q_b = 30
        Q_c = 50
        Di_a = 40
        Di_b = 5
        Di_c = 1

    base_path = path
    input_file_path = os.path.join(base_path, f"{prefix}", f"filtered_footprints_{prefix}.csv")

    columns = ['Za', 'Zb', 'Zc', 'OrdersHandled']
    result_columns = ['Xa', 'Xb', 'Xc', 'OrdersHandled']
    calculated_data = []
    specific_point = None

    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Xa = float(row['Za']) * Q_a * Di_a / 240
            Xb = float(row['Zb']) * Q_b * Di_b / 240
            Xc = float(row['Zc']) * Q_c * Di_c / 240
            OrdersHandled = int(row['OrdersHandled'])
            W2 = float(row['W2'])
            W3 = float(row['W3'])
            calculated_data.append({
                'Xa': Xa,
                'Xb': Xb,
                'Xc': Xc,
                'OrdersHandled': OrdersHandled
            })
            if round(W2, 1) == 1 and W3 == 1:
                specific_point = (Xa, Xb, Xc)

    #calculated_data = sorted(calculated_data, key=lambda x: x['OrdersHandled'])

    grouped_data = defaultdict(list)
    for data in calculated_data:
        key = (data['Xa'], data['Xb'], data['Xc'])
        grouped_data[key].append(data['OrdersHandled'])

    avg_data = []
    for key, orders in grouped_data.items():
        avg_orders_handled = sum(orders) / len(orders)
        avg_data.append({
            'Xa': key[0],
            'Xb': key[1],
            'Xc': key[2],
            'OrdersHandled': avg_orders_handled
        })

    calculated_data = avg_data
    calculated_data = sorted(calculated_data, key=lambda x: x['OrdersHandled'])

    output_file_path = os.path.join(base_path, f"{prefix}", f"calculated_values_{prefix}.csv")
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_columns)
        writer.writeheader()
        writer.writerows(calculated_data)



    scale = 100  # Scale of the ternary plot
    # Prepare the data for the ternary plot
    points = []
    orders_handled = []

    for data in calculated_data:
        Xa = data['Xa']
        Xb = data['Xb']
        Xc = data['Xc']
        OrdersHandled = data['OrdersHandled']

        # Normalize the coordinates so that Xa + Xb + Xc = scale
        total = Xa + Xb + Xc
        if total > 0:
            points.append((Xa / total * scale, Xb / total * scale, Xc / total * scale))
            orders_handled.append(OrdersHandled)

    if specific_point:
        Xa, Xb, Xc = specific_point
        total = Xa + Xb + Xc
        specific_point = (Xa / total * scale, Xb / total * scale, Xc / total * scale)

    # Create the ternary plot
    fig, ax = plt.subplots(figsize=(8, 8))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    # Draw the boundary and gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="grey")

    # Set title and axis labels

    #tax.set_title(f"Pods/SKU per class for demand {prefix}", fontsize=20)
    tax.set_title(f"Pods/SKU per class for peak Gall&Gall demand", fontsize=20)
    tax.left_axis_label("C", fontsize=20, offset=0.15)
    tax.right_axis_label("B", fontsize=20, offset=0.15)
    tax.bottom_axis_label("A", fontsize=20, offset=0.1)

    # Compute ticks based on X values
    tick_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tick_values_A = [tick * 2.4 / Q_a for tick in tick_steps]
    tick_values_B = [tick * 2.4/ Q_b for tick in tick_steps]
    tick_values_C = [tick * 2.4/ Q_c for tick in tick_steps]

    offset_value = 0.02
    tax.ticks(axis='l', ticks=tick_values_C, tick_formats="%.1f",fontsize=13, offset=offset_value)
    tax.ticks(axis='r', ticks=tick_values_B, tick_formats="%.1f",fontsize=13, offset=offset_value+0.007)
    tax.ticks(axis='b', ticks=tick_values_A, tick_formats="%.1f",fontsize=13, offset=offset_value)



    # Define colormap and normalization
    norm = Normalize(vmin=min(orders_handled), vmax=max(orders_handled))
    cmap = plt.cm.get_cmap('plasma')

    # Plot each point with corresponding color
    for point, oh in zip(points, orders_handled):
        color = cmap(norm(oh))
        size = 100
        if (point[0] == specific_point[0] and point[1] == specific_point[1] and point[2] == specific_point[2]):
            size += 300
        tax.scatter([point], color=color, marker='x', s=size, zorder=3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Orders Handled', fontsize=20)
    cbar.ax.tick_params(labelsize=13)

    #tax.ticks(axis='lbr', multiple=10, fontsize=10)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    plt.savefig(os.path.join(os.path.join(base_path, f"{prefix}"), f"ternary_plot_{prefix}.png"), bbox_inches='tight')
    plt.show()
    plt.close(fig)

def create_ternary_plot_items_pod(prefix, path):
    if prefix == 'Da':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 58
        Di_b = 10
        Di_c = 4
    if prefix == 'Db':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 34
        Di_b = 5
        Di_c = 3
    if prefix == 'Dc':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 69
        Di_b = 7
        Di_c = 3
    if prefix == 'Dd':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 40
        Di_b = 3
        Di_c = 2
    if prefix == 'De':
        Q_a = 10
        Q_b = 10
        Q_c = 80
        Di_a = 86
        Di_b = 6
        Di_c = 1
    if prefix == 'Df':
        Q_a = 20
        Q_b = 40
        Q_c = 40
        Di_a = 44
        Di_b = 2
        Di_c = 1
    if prefix == 'Dr':
        Q_a = 20
        Q_b = 30
        Q_c = 50
        Di_a = 37
        Di_b = 7
        Di_c = 1
    if prefix == 'Dp':
        Q_a = 20
        Q_b = 30
        Q_c = 50
        Di_a = 40
        Di_b = 5
        Di_c = 1

    base_path = path
    input_file_path = os.path.join(base_path, f"{prefix}", f"filtered_footprints_{prefix}.csv")

    columns = ['Za', 'Zb', 'Zc', 'OrdersHandled']
    result_columns = ['Xa', 'Xb', 'Xc', 'OrdersHandled']
    calculated_data = []

    with open(input_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Xa = float(row['Za']) * Q_a * Di_a / 240
            Xb = float(row['Zb']) * Q_b * Di_b / 240
            Xc = float(row['Zc']) * Q_c * Di_c / 240
            OrdersHandled = int(row['OrdersHandled'])
            calculated_data.append({
                'Xa': Xa,
                'Xb': Xb,
                'Xc': Xc,
                'OrdersHandled': OrdersHandled
            })

    #calculated_data = sorted(calculated_data, key=lambda x: x['OrdersHandled'])

    grouped_data = defaultdict(list)
    for data in calculated_data:
        key = (data['Xa'], data['Xb'], data['Xc'])
        grouped_data[key].append(data['OrdersHandled'])

    avg_data = []
    for key, orders in grouped_data.items():
        avg_orders_handled = sum(orders) / len(orders)
        avg_data.append({
            'Xa': key[0],
            'Xb': key[1],
            'Xc': key[2],
            'OrdersHandled': avg_orders_handled
        })

    calculated_data = avg_data
    #calculated_data = sorted(calculated_data, key=lambda x: x['OrdersHandled'])

    output_file_path = os.path.join(base_path, f"{prefix}", f"calculated_values_{prefix}.csv")
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_columns)
        writer.writeheader()
        writer.writerows(calculated_data)



    scale = 100  # Scale of the ternary plot
    # Prepare the data for the ternary plot
    points = []
    orders_handled = []

    for data in calculated_data:
        Xa = data['Xa']
        Xb = data['Xb']
        Xc = data['Xc']
        OrdersHandled = data['OrdersHandled']

        # Normalize the coordinates so that Xa + Xb + Xc = scale
        total = Xa + Xb + Xc
        if total > 0:
            points.append((Xa / total * scale, Xb / total * scale, Xc / total * scale))
            orders_handled.append(OrdersHandled)

    # Create the ternary plot
    fig, ax = plt.subplots(figsize=(8, 8))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    # Draw the boundary and gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="grey")

    # Set title and axis labels

    tax.set_title(f"Pods/SKU per class for peak Gall&Gall demand", fontsize=20)
    tax.left_axis_label("C", fontsize=20, offset=0.15)
    tax.right_axis_label("B", fontsize=20, offset=0.15)
    tax.bottom_axis_label("A", fontsize=20, offset=0.1)

    # Compute ticks based on X values
    tick_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tick_values_A = [tick * 2.4 / Q_a for tick in tick_steps]
    tick_values_B = [tick * 2.4/ Q_b for tick in tick_steps]
    tick_values_C = [tick * 2.4/ Q_c for tick in tick_steps]

    offset_value = 0.02
    tax.ticks(axis='l', ticks=tick_values_C, tick_formats="%.1f",fontsize=13, offset=offset_value)
    tax.ticks(axis='r', ticks=tick_values_B, tick_formats="%.1f",fontsize=13, offset=offset_value+0.007)
    tax.ticks(axis='b', ticks=tick_values_A, tick_formats="%.1f",fontsize=13, offset=offset_value)



    # Define colormap and normalization
    norm = Normalize(vmin=min(orders_handled), vmax=max(orders_handled))
    cmap = plt.cm.get_cmap('plasma')

    # Plot each point with corresponding color
    for point, oh in zip(points, orders_handled):
        color = cmap(norm(oh))
        tax.scatter([point], color=color, marker='x', s=100, zorder=3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Orders Handled', fontsize=20)
    cbar.ax.tick_params(labelsize=13)

    #tax.ticks(axis='lbr', multiple=10, fontsize=10)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    plt.savefig(os.path.join(os.path.join(base_path, f"{prefix}"), f"ternary_plot_{prefix}.png"), bbox_inches='tight')
    plt.show()
    plt.close(fig)



base_path = r'C:\Users\pnl0j327\RAWSim-O\RAWSim-O-main\Material\Instances\10.Weighted30min Demand Gall'
#base_path = r'C:\Users\pnl0j327\RAWSim-O\RAWSim-O-main\Material\Instances\0.Weighted30min Demand separate'
prefix = 'Dp'

read_footprints(prefix, base_path)
write_avg_data(prefix, base_path)

create_ternary_plot(prefix, base_path)
