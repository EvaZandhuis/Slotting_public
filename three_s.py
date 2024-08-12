import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import csv
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def cD(i, s):
    return i ** (s)

def D(i, s):
    return s * i ** (s-1)

i_values = np.linspace(0.0001, 1, 1000)


s_values = [0.222, 0.139, 0.065]
colors = ['g', 'orange', 'r']
demand = 1000
SKUS = 100
# the separation of classes is at these i-axis ticks
class_i1 = 0.2
class_i2 = 0.6
data = []

plt.figure(figsize=(10, 6))
for s, color in zip(s_values, colors):
    #plt.plot(i_values, cD(i_values, s), label=f's={s}', color=color)
    plt.plot(i_values, D(i_values, s), color=color, linestyle='dashed')

plt.title('Plot of G(i) and g(i) with different values of s')
plt.xlabel('Assortment (i) [%]')
plt.ylabel('Demand [%]')
plt.ylim(0, 1)
plt.xlim(0,1)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.savefig(f'3s_cD_D.png')
plt.show()

#cumulative + bins
plt.figure(figsize=(10, 6))
for s, color in zip(s_values, colors):
    plt.plot(i_values, cD(i_values, s), label=f's={s}', color=color)
    #plt.plot(0.2, cD(0.2, s), 'o', color=color)
    #plt.annotate(f'({0.2}, {cD(0.2, s):.1f})', (0.2, cD(0.2, s)), textcoords="offset points", xytext=(0, 10), ha='center',color=color)

    block_areas = []
    stepsize = 0.1
    for i in np.arange(0, 1, stepsize):
        #block_height = np.mean(cD(i_values[np.logical_and(i_values >= i, i_values < i + stepsize)], s))
        block_height = cD(i+0.1, s)
        block_area = stepsize * block_height
        block_areas.append(block_area)
        plt.gca().add_patch(plt.Rectangle((i, 0), stepsize, block_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1), edgecolor=color, linestyle='dashed'))
    total_area = sum(block_areas)
    #print(f"For s={s}, Areas of bins: {block_areas}")
    #print(f"Total area for s={s}: {total_area}")

plt.title('Plot of cumulative G(i) with right Riemann sum with different values of s')
plt.xlabel('Assortment (i) [%]')
plt.ylabel('Demand [%]')
plt.ylim(0, 1)
plt.xlim(0,1)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.close()
#plt.savefig('3s_BinsCumulativeDemand.png')
plt.show()

#cumulative + classes (bins)
plt.figure(figsize=(6, 6))
for s, color in zip(s_values, colors):
    #plt.plot(i_values, cD(i_values, s), label=f's={s}', color=color)
    plt.plot(i_values, D(i_values, s), color=color, label=f's={s}', linestyle='dashed')
    #plt.plot(0.2, cD(0.2, s), 'o', color=color)
    #plt.annotate(f'({0.2}, {cD(0.2, s):.1f})', (0.2, cD(0.2, s)), textcoords="offset points", xytext=(0, 10), ha='center',color=color)

    #block_areas = []
    #stepsize = 0.1
    #for i in np.arange(0, 1, stepsize):
        #block_height = np.mean(cD(i_values[np.logical_and(i_values >= i, i_values < i + stepsize)], s))
    block1_height = cD(class_i1, s)
    block1_width = class_i1
    block2_height = cD(class_i2, s) - block1_height
    block2_width = class_i2 - class_i1
    block3_height = cD(1, s) - cD(class_i2, s)
    block3_width = 1 - class_i2

    plt.gca().add_patch(plt.Rectangle((0, 0), block1_width, block1_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))
    plt.gca().add_patch(plt.Rectangle((class_i1, 0), block2_width, block2_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))
    plt.gca().add_patch(plt.Rectangle((class_i2, 0), block3_width, block3_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))

    area_block1 = block1_width * block1_height
    area_block2 = block2_width * block2_height
    area_block3 = block3_width * block3_height
    #total_area = area_block1 + area_block2 + area_block3
    #print('1', block1_height , block2_height, block3_height)

plt.title('Plot of Bins in G(i) with different values of s')
plt.xlabel('Assortment (i) [%]')
plt.ylabel('Demand [%]')
plt.ylim(0, 1)
plt.xlim(0,1)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.close()
#plt.savefig(f'3s_BinsClass{class_i1}{class_i2}.png')
plt.show()


#plot of demand sku distribution
def demand_sku_distribution(SKUS, demand, s_values, colors, class_i1, class_i2, filepath):
    data = []
    plt.figure(figsize=(6, 6))
    for s, color in zip(s_values, colors):
        # plt.plot(i_values, D(i_values, s), label=f's={s}', color=color)
        # plt.plot(0.2, cD(0.2, s), 'o', color=color)
        # plt.annotate(f'({0.2}, {cD(0.2, s):.1f})', (0.2, cD(0.2, s)), textcoords="offset points", xytext=(0, 10), ha='center',color=color)

        # block_areas = []
        # stepsize = 0.1
        # for i in np.arange(0, 1, stepsize):
        # block_height = np.mean(cD(i_values[np.logical_and(i_values >= i, i_values < i + stepsize)], s))
        block1_height = cD(class_i1, s)
        block1_width = class_i1
        block2_height = cD(class_i2, s) - block1_height
        block2_width = class_i2 - class_i1
        block3_height = cD(1, s) - cD(class_i2, s)
        block3_width = 1 - class_i2

        # plt.gca().add_patch(plt.Rectangle((0, 0), block1_width, block1_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))
        # plt.gca().add_patch(plt.Rectangle((class_i1, 0), block2_width, block2_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))
        # plt.gca().add_patch(plt.Rectangle((class_i2, 0), block3_width, block3_height, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.1),edgecolor=color, linestyle='dashed'))

        area_block1 = block1_width * block1_height
        area_block2 = block2_width * block2_height
        area_block3 = block3_width * block3_height
        # print('1', block1_height , block2_height, block3_height)

        skus_classA = round(block1_width * SKUS)
        skus_classB = round(block2_width * SKUS)
        skus_classC = round(block3_width * SKUS)
        demand_classA = block1_height * demand
        demand_classB = block2_height * demand
        demand_classC = block3_height * demand
        sku_demand_classA = demand_classA / skus_classA
        sku_demand_classB = demand_classB / skus_classB
        sku_demand_classC = demand_classC / skus_classC

        cumulative_demand = 0

        for sku in np.arange(0, skus_classA, 1):
            #if s == 0.222:
            #    low = 0
            #if s == 0.139:
            #    low = 35
            #if s == 0.065:
            #    low = 40
            plt.gca().add_patch(plt.Rectangle(((1 / SKUS) * sku, 0), 1 / SKUS, sku_demand_classA, fill=True, facecolor=mcolors.to_rgba(color, alpha=0.2)))
            #plt.gca().add_patch(plt.Rectangle(((1 / SKUS) * sku, low), 1 / SKUS, sku_demand_classA - low, fill=True,facecolor=mcolors.to_rgba(color, alpha=0.3)))
            cumulative_demand += sku_demand_classA / demand
            data.append(
                {'s': s, 'SKU': sku, 'Demand': sku_demand_classA, '% of Total Demand': sku_demand_classA / demand,
                 '% Cumulative Demand': cumulative_demand})
        for sku in np.arange(0, skus_classB, 1):
            plt.gca().add_patch(plt.Rectangle(((1 / SKUS) * sku + class_i1, 0), 1 / SKUS, sku_demand_classB, fill=True,
                                              facecolor=mcolors.to_rgba(color, alpha=0.3)))
            cumulative_demand += sku_demand_classB / demand
            data.append({'s': s, 'SKU': sku + (class_i1 * SKUS), 'Demand': sku_demand_classB,
                         '% of Total Demand': sku_demand_classB / demand, '% Cumulative Demand': cumulative_demand})
        for sku in np.arange(0, skus_classC, 1):
            plt.gca().add_patch(plt.Rectangle(((1 / SKUS) * sku + class_i2, 0), 1 / SKUS, sku_demand_classC, fill=True,
                                              facecolor=mcolors.to_rgba(color, alpha=0.3)))
            cumulative_demand += sku_demand_classC / demand
            data.append({'s': s, 'SKU': sku + (class_i2 * SKUS), 'Demand': sku_demand_classC,
                         '% of Total Demand': sku_demand_classC / demand, '% Cumulative Demand': cumulative_demand})

    # data.append({'s': s, 'SKU': i, 'Demand': block_demand, '% of Total Demand': block_demand/demand,'% Cumulative Demand': cumulative_demand})
    #

    ## Print data to CSV file
    csv_file_name = f'Demand_values_{class_i1}-{class_i2}SKUs{SKUS}_Demand{demand}.csv'
    if filepath is None:
        with open(csv_file_name, "w", newline='') as csvfile:
            fieldnames = ['s', 'SKU', 'Demand', '% of Total Demand', '% Cumulative Demand']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    else:
        with open(os.path.join(filepath, csv_file_name), "w", newline='') as csvfile:
            fieldnames = ['s', 'SKU', 'Demand', '% of Total Demand', '% Cumulative Demand']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    plt.title('Plot of SKUs in classes for g(i) with different values of s')
    plt.xlabel('SKUs [i]')
    plt.ylabel('Demand [items]')
    plt.ylim(0, sku_demand_classA + sku_demand_classB)
    # plt.ylim(0,100)
    plt.xlim(0, 1)
    ticks = [0.2, 0.4, 0.6, 0.8, 1]
    plt.xticks(ticks, [int(tick * SKUS) for tick in ticks])
    plt.legend(handles=[mpatches.Patch(color=color, label=f's={s}') for s, color in zip(s_values, colors)])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if filepath is None:
        plt.savefig(f'3s_SKUs{SKUS}Class{class_i1}-{class_i2}Demand{demand}.png')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, f'3s_SKUs{SKUS}Class{class_i1}-{class_i2}Demand{demand}.png'))
    #plt.close()
    plt.show()

demand_sku_distribution(SKUS, demand, s_values, colors, class_i1, class_i2, None)

