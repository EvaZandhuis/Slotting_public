import os
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def read_data(directory):
    demand_data = {}
    file_pattern = re.compile(r'demand_D(r|p)_overview\.')

    for filename in os.listdir(directory):
        if file_pattern.match(filename):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                for line in reader:
                    demand_name = line[0]
                    pods_per_item = eval(line[2])
                    items_per_pod = eval(line[3])
                    pods_per_sku = eval(line[4])
                    if demand_name not in demand_data:
                        demand_data[demand_name] = {
                            'pods_per_item': [[], [], []],
                            'items_per_pod': [[], [], []],
                            'pods_per_sku': [[], [], []]
                        }
                    for i in range(3):
                        demand_data[demand_name]['pods_per_item'][i].append(pods_per_item[i])
                        demand_data[demand_name]['items_per_pod'][i].append(items_per_pod[i])
                        demand_data[demand_name]['pods_per_sku'][i].append(pods_per_sku[i])

    sorted_demand_data = {k: demand_data[k] for k in sorted(demand_data.keys(), reverse=True)}

    return sorted_demand_data


def plot_metric_violin(metric_data, title, colors, labels, demand_names):
    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(metric_data, showmeans=False, showmedians=True)

    # Set the title and labels
    ax.set_title(f'{title} for classes per Gall&Gall demand profile')
    ax.set_ylabel(title)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Assign different colors to each demand class and set transparency

    # Assign different colors to each demand class and set transparency for fills
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i // 3])
        pc.set_edgecolor(colors[i // 3])
        if i % 3 == 1:  # Second violin for all demands
            pc.set_alpha(0.6)  # 50% transparency
        elif i % 3 == 2:  # Third violin for all demands
            pc.set_alpha(0.4)  # 80% transparency
        else:  # First violin for all demands
            pc.set_alpha(1)  # No transparency

    # Ensure the medians remain opaque
    for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_alpha(1)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a custom legend
    demand_names = ['Regular', 'Peak']
    legend_handles = [mpatches.Patch(color=colors[i], label=demand_names[i]) for i in range(len(demand_names))]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{title}_violinplot_gall.png'))
    plt.show()
    plt.close(fig)


def plot_metric_box(metric_data, title, colors, labels, demand_names):
    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(metric_data, patch_artist=True)
    #box = ax.boxplot(metric_data, whis=100, patch_artist=True)

    # Set the title and labels
    ax.set_title(f'Range of {title} for classes per demand profile')
    ax.set_ylabel(title)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Assign different colors to each demand class and set transparency

    for i, patch in enumerate(box['boxes']):
        if i % 3 == 1:  # Second boxplot for all demands
            patch.set_facecolor(colors[i // 3])
            patch.set_alpha(0.6)  # 50% transparency
        elif i % 3 == 2:  # Third boxplot for all demands
            patch.set_facecolor(colors[i // 3])
            patch.set_alpha(0.4)  # 80% transparency
        else:  # First boxplot for all demands
            patch.set_facecolor(colors[i // 3])
            patch.set_alpha(1)  # No transparency

    for component in ['medians', 'whiskers', 'caps', 'fliers']:
        plt.setp(box[component], color='black')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a custom legend
    legend_handles = [mpatches.Patch(color=colors[i], label=demand_names[i]) for i in range(len(demand_names))]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{title}_boxplot.png'))
    plt.show()
    plt.close(fig)

    box_stats = {
        'Demand' : [],
        'Label': [],
        'Mean': [],
        'Median': [],
        'Lower quantile': [],
        'Upper quantile': [],
        'Lower whisker': [],
        'Upper whisker': []
    }

    for i, label in enumerate(labels):
        demand_index = i // 3
        data = metric_data[i]
        lower_whisker = box['whiskers'][2 * i].get_ydata()[1]
        upper_whisker = box['whiskers'][2 * i + 1].get_ydata()[1]
        median = box['medians'][i].get_ydata()[0]
        q1 = box['boxes'][i].get_path().vertices[1][1]
        q3 = box['boxes'][i].get_path().vertices[2][1]

        box_stats['Demand'].append(demand_names[demand_index])
        box_stats['Label'].append(label)
        box_stats['Mean'].append(round(np.mean(data), 3))
        box_stats['Median'].append(round(median, 3))
        box_stats['Lower quantile'].append(round(q1, 3))
        box_stats['Upper quantile'].append(round(q3, 3))
        box_stats['Lower whisker'].append(round(lower_whisker, 3))
        box_stats['Upper whisker'].append(round(upper_whisker, 3))

        # Write statistics to CSV
    csv_filename = os.path.join(directory, f'data_plots_overview_{title}.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(box_stats.keys())
        csvwriter.writerows(zip(*box_stats.values()))






def plot_data(demand_data):
    """
    Plots boxplots for pods/item, items/pod, and pods/sku from demand data.

    Args:
    - demand_data (dict): A dictionary with demand names as keys and lists of [pods/item, items/pod, pods/sku] as values.
    """
    # Prepare data for boxplots
    pods_per_item = []
    items_per_pod = []
    pods_per_sku = []
    labels = []

    demand_names = list(demand_data.keys())
    #colors = ['r', 'g', 'b', 'c', 'm', 'orange'][:len(demand_names)]  # Extend or reduce colors as needed
    colors = ['r', 'orange', 'yellow'][:len(demand_names)]

    for demand in demand_names:
        for i in range(3):
            pods_per_item.append(demand_data[demand]['pods_per_item'][i])
            items_per_pod.append(demand_data[demand]['items_per_pod'][i])
            pods_per_sku.append(demand_data[demand]['pods_per_sku'][i])
            labels.append(['A', 'B', 'C'][i])

    # Plot each metric separately
    plot_metric_box(pods_per_item, 'Pods per Item', colors, labels, demand_names)
    plot_metric_box(items_per_pod, 'Items per Pod', colors, labels, demand_names)
    plot_metric_box(pods_per_sku, 'Pods per SKU', colors, labels, demand_names)

    plot_metric_violin(pods_per_item, 'Pods per Item', colors, labels, demand_names)
    plot_metric_violin(items_per_pod, 'Items per Pod', colors, labels, demand_names)
    plot_metric_violin(pods_per_sku, 'Pods per SKU', colors, labels, demand_names)

plt.rcParams.update({'font.size': 20,  # controls default text sizes
		                     'axes.titlesize': 20,  # fontsize of the axes title
		                     'axes.labelsize': 20,  # fontsize of the x and y labels
		                     'xtick.labelsize': 20,  # fontsize of the tick labels
		                     'ytick.labelsize': 20,  # fontsize of the tick labels
		                     'legend.fontsize': 20,  # fontsize of the legend
		                     'figure.titlesize': 20  # fontsize of the figure title
		                     })

# Example usage:
directory = r'C:\Users\pnl0j327\PycharmProjects\pythonProject1\Gall_step12_S100_d1000_T120'
demand_data = read_data(directory)
plot_data(demand_data)