import matplotlib.pyplot as plt
import numpy as np

# Data for Chou and Weidinger
weidinger_data = ([20, 30, 50], [80, 15, 5], 'Weidinger, 2019', 'blue')
chou_data = ([10, 30, 60], [60, 25, 15], 'Chou, 2019', 'skyblue')
datasets = [chou_data, weidinger_data]


# Setup the figure and axes for the fourth plot
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Reuse the existing demand curves data
s_values = [0.318, 0.222, 0.139, 0.065]
s2_values = [1, 0.748, 0.569, 0.431]
i_values = np.linspace(0.01, 1, 100)
scaled_i_values = i_values * 100
colors = ['y', 'g', 'orange', 'r']  # blue, green, red, cyan
colors2 = ['lightgreen', 'pink', 'mediumpurple', 'brown']

# Plot the demand curves and mark the point at 20%
for s, color in zip(s_values, colors):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax4.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    G_at_20 = int(round((0.2 ** s) * 100))
    ax4.plot(20, G_at_20, 'o', color=color)
    ax4.annotate(f'({20}, {G_at_20})', (20, G_at_20), textcoords="offset points", xytext=(0, 10), ha='center', color=color)

for s, color in zip(s2_values, colors2):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax4.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    G_at_20 = int(round((0.2 ** s) * 100))
    ax4.plot(20, G_at_20, 'o', color=color)  # Dot at x = 20%
    ax4.annotate(f'({20}, {G_at_20:.1f})', (20, G_at_20), textcoords="offset points", xytext=(10, -17), ha='center', color=color)

# Plot the cumulative demand curves for Chou and Weidinger
for data in datasets:
    assortment_percent, demand_percent, label, line_color = data
    widths = np.array(assortment_percent)
    cumulative_widths = np.cumsum(widths)
    cumulative_demands = np.cumsum(demand_percent)
    ax4.plot(cumulative_widths, cumulative_demands, 'o-', label=label, color=line_color)
    for x, y in zip(cumulative_widths, cumulative_demands):
        if (x, y) != (100, 100):
            ax4.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(-24, -4), ha='center', color=line_color)

# exp function
x1 = np.linspace(0.01, 1, 1000)
cdf1 = x1 - x1 * np.log(x1)
plt.plot(x1*100, cdf1*100, label='Exponential function', color='purple')
y_at_20 = round((0.2 - 0.2 * np.log(0.2))*100)
ax4.plot(20, y_at_20, 'o', color='purple')
ax4.annotate(f'({20}, {y_at_20})', (20, y_at_20), textcoords="offset points", xytext=(-10, 10), ha='center', color='purple')

# Customize the plot
ax4.set_xlabel('% of Assortment')
ax4.set_ylabel('% of Demand')
ax4.set_title('Combined Demand Curves from literature')
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
ax4.set_xticks(np.arange(0, 101, 10))
ax4.set_yticks(np.arange(0, 101, 10))
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot
plt.tight_layout()
plt.savefig('Combined_Demand_Curves.png')
plt.show()
plt.close(fig4)
