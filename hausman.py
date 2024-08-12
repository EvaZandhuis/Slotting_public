import matplotlib.pyplot as plt
import numpy as np

# Define the values of s
s_values = [0.318, 0.222, 0.139, 0.065]

# Define the range for i, from 0.01 to 1, scaled by 100 (1 to 100)
i_values = np.linspace(0.01, 1, 100)
scaled_i_values = i_values * 100  # Scale for the plot

# Define a list of colors for the plots
colors = ['y', 'g', 'orange', 'r']  # blue, green, red, cyan

# Create the first plot with only demand curves
fig1, ax1 = plt.subplots(figsize=(10, 6))
for s, color in zip(s_values, colors):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax1.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    G_at_20 = int(round((0.2 ** s) * 100))
    ax1.plot(20, G_at_20, 'o', color=color)
    ax1.annotate(f'({20}, {G_at_20:.1f})', (20, G_at_20), textcoords="offset points", xytext=(0, 10), ha='center')

ax1.set_xlabel('% of Assortment')
ax1.set_ylabel('% of Demand')
ax1.set_title('Demand Curves (Hausman et al., 1976)')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.set_xticks(np.arange(0, 101, 10))
ax1.set_yticks(np.arange(0, 101, 10))
ax1.legend()
plt.tight_layout()
plt.savefig('Demand_Curves_Only.png')
plt.show()
plt.close(fig1)

# Create the second plot: Demand curves with vertical lines
fig2, ax2 = plt.subplots(figsize=(10, 6))
line_positions = [16.6, 12.6, 8, 3.3]
for index, (s, color) in enumerate(zip(s_values, colors)):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax2.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    x_line = line_positions[index]
    idx = np.abs(scaled_i_values - x_line).argmin()
    y_intersect = scaled_G_i[idx]
    ax2.vlines(x=x_line, ymin=0, ymax=y_intersect, color=color, linestyle='--', label=f'{x_line:.2f} % of assortment')
    print(f'Intersection at {x_line}% of assortment (Color {color}): ({x_line:.2f}, {y_intersect:.1f})')

ax2.set_xlabel('% of Assortment')
ax2.set_ylabel('% of Demand')
ax2.set_title('Demand Curves with two classes (Hausman et al., 1976)')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.set_xticks(np.arange(0, 101, 10))
ax2.set_yticks(np.arange(0, 101, 10))
#ax2.legend()
plt.tight_layout()
plt.savefig('Demand_Curves_with_Vertical_Lines_and_Intersections.png')
plt.show()
plt.close(fig2)

# Create the third plot: Demand curves with multiple vertical lines
fig3, ax3 = plt.subplots(figsize=(10, 6))
line_positions = {
    'y': [1, 33],
    'g': [1, 27.5],
    'orange': [0.56, 20],
    'r': [0.25, 16]
}
for s, color in zip(s_values, colors):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax3.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    for x_line in line_positions[color]:
        idx = np.abs(scaled_i_values - x_line).argmin()
        y_intersect = scaled_G_i[idx]
        ax3.vlines(x=x_line, ymin=0, ymax=y_intersect, color=color, linestyle='--', label=f'{x_line:.2f} % of assortment')
        print(f'Intersection at {x_line}% of assortment (Color {color}): ({x_line:.2f}, {y_intersect:.1f})')

ax3.set_xlabel('% of Assortment')
ax3.set_ylabel('% of Demand')
ax3.set_title('Detailed Demand Curves with three classes (Hausman et al., 1976)')
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.set_xticks(np.arange(0, 101, 10))
ax3.set_yticks(np.arange(0, 101, 10))
#.legend()
plt.tight_layout()
plt.savefig('Demand_Curves_with_Multiple_Vertical_Lines_and_Intersections.png')
plt.show()
plt.close(fig3)






