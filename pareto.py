import matplotlib.pyplot as plt
import numpy as np

def create_plot(assortment_percent, demand_percent, title, colors, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert assortment percentages into proportions for the plot
    widths = np.array(assortment_percent)
    # Calculate cumulative widths for positioning and labeling
    cumulative_widths = np.cumsum(widths)
    cumulative_demands = np.cumsum(demand_percent)

    # Create the bar plot with named classes
    x_pos = 0  # Starting x position for the first bar
    for i, width in enumerate(widths):
        ax.bar(x_pos, demand_percent[i], width=width, align='edge',
               color=colors[i], label=f'Class {chr(65+i)}')
        x_pos += width

    # Plotting the demand curve line
    ax.plot(cumulative_widths, cumulative_demands, 'o-', color=colors[-1], label='Demand Curve')

    # Setting labels and titles
    ax.set_xlabel('% of Total Assortment')
    ax.set_ylabel('% Total Demand')
    ax.set_title('Distribution of Classes ' + title)

    # Setting limits and ticks for better readability
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))

    # Adding a legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    plt.close()

def create_comparison_plot(datasets, titles, colors, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting the demand curve comparisons
    for data, title, color in zip(datasets, titles, colors):
        assortment_percent, demand_percent = data
        widths = np.array(assortment_percent)
        cumulative_widths = np.cumsum(widths)
        cumulative_demands = np.cumsum(demand_percent)
        ax.plot(cumulative_widths, cumulative_demands, 'o-', label=title, color=color)

    ax.set_title('Comparison of Demand Curves')
    ax.set_xlabel('% of Total Assortment')
    ax.set_ylabel('% Total Demand')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Colors for plots
colors1 = ['red', 'orange', 'yellow', 'blue']
colors2 = ['red', 'orange', 'yellow', 'skyblue']

# Data sets
data1 = ([10, 30, 60], [60, 25, 15])
data2 = ([20, 30, 50], [80, 15, 5])
datasets = [data1, data2]
titles = ['(Chou, 2019)', '(Weidinger, 2019)']
colors = [colors1[-1], colors2[-1]]  # Using last color for lines

# Create plots and save them
create_plot(*data1, '(Chou, 2019)', colors1, 'Chou_2019.png')
create_plot(*data2, '(Weidinger, 2019)', colors2, 'Weidinger_2019.png')
create_plot(*data2, '(Weidinger, 2019)', colors2, 'Weidinger_2019.png')
#create_comparison_plot(datasets, titles, colors, 'Comparison_of_Demand_Curves.png')



