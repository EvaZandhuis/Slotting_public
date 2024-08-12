import matplotlib.pyplot as plt
import numpy as np

# Define the new values of s
s_values = [1, 0.748, 0.569, 0.431]

# Define the range for i, from 0.01 to 1, scaled by 100 (1 to 100)
i_values = np.linspace(0.01, 1, 100)
scaled_i_values = i_values * 100  # Scale for the plot

# Define a list of colors for the plots
colors = ['lightgreen', 'pink', 'mediumpurple', 'brown']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
for s, color in zip(s_values, colors):
    G_i = i_values ** s
    scaled_G_i = G_i * 100
    ax.plot(scaled_i_values, scaled_G_i, label=f'G(i) = i^{s:.3f}', color=color)
    G_at_20 = int(round((0.2 ** s) * 100))
    ax.plot(20, G_at_20, 'o', color=color)  # Dot at x = 20%
    ax.annotate(f'({20}, {G_at_20:.1f})', (20, G_at_20), textcoords="offset points", xytext=(0, 10), ha='center', color='black')  # Coordinates for x = 20%

ax.set_xlabel('% of Assortment')
ax.set_ylabel('% of Demand')
ax.set_title('Demand Curves (Guo, 2016)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_yticks(np.arange(0, 101, 10))
ax.legend()
plt.tight_layout()
plt.savefig('Demand_Curves_New_S_Values.png')
plt.show()
