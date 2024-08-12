import pandas as pd
import random
import matplotlib.pyplot as plt

# Manual input parameters
number_of_orders = 60
number_of_skus = 10
total_items = 124

#number of skus per turnover rate
l_skus = 0.2 * number_of_skus
m_skus = 0.4 * number_of_skus
s_skus = 0.4 * number_of_skus

#percentage van totaal flessen
p_l = 0.65
p_m = 0.3
p_s = 0.05


# Generate SKUs with specified amounts
def generate_skus(total_items):
    skus = {}

    for i in range(int(l_skus)): #voor alle skus in categorie l,
        sku_amount = int(p_l * total_items / l_skus)
        skus[f"SKU{i + 1}"] = sku_amount

    for i in range(int(l_skus), int(m_skus+l_skus)):
        sku_amount = int(p_m * total_items / m_skus)
        skus[f"SKU{i + 1}"] = sku_amount

    for i in range(int(m_skus+l_skus), number_of_skus):
        sku_amount = int(p_s * total_items / s_skus)
        skus[f"SKU{i + 1}"] = sku_amount

    return skus


def distribute_items(skus, number_of_orders):
    orders = [{} for _ in range(number_of_orders)]
    sku_list = []

    for sku, amount in skus.items():
        sku_list.extend([sku] * amount)

    random.shuffle(sku_list)

    current_order_index = 0

    while sku_list:
        sku = sku_list.pop(0)  # Take the first sku from the list

        # Add the sku to the current order
        orders[current_order_index][sku] = orders[current_order_index].get(sku, 0) + 1

        current_order_index = (current_order_index + 1) % number_of_orders  # Move to the next order

    return orders


def visualize_sku_distribution(orders):
    sku_quantities = {f"SKU{i}": [0] * len(orders) for i in range(1, len(skus) + 1)}

    for i, order in enumerate(orders):
        for sku, quantity in order.items():
            sku_quantities[sku][i] = quantity

    for order_num, order in enumerate(orders, start=1):
        print(f"Order {order_num}: {order}")

    # Define colors inside the function
    colors = [plt.cm.Paired(i / len(skus)) for i in range(len(skus))]

    fig, ax = plt.subplots()

    for i, sku in enumerate(skus.keys()):
        # Initialize a list to store the cumulative bottom values
        bottom_values = [0] * len(orders)

        # Convert dict_keys to list for subscripting
        sku_keys_list = list(skus.keys())

        # Iterate through previous SKUs and accumulate their quantities
        for j in range(i):
            for order_num in range(len(orders)):
                bottom_values[order_num] += sku_quantities[sku_keys_list[j]][order_num]

        # Plot the bar with the accumulated values as the bottom parameter
        ax.bar(range(1, len(orders) + 1), sku_quantities[sku], label=sku, color=colors[i], bottom=bottom_values)

    ax.set_xlabel('Order')
    ax.set_ylabel('Quantity')
    ax.set_title('SKU Distribution Across Orders')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

# Generate SKUs, distribute items, and visualize SKU distribution
skus = generate_skus(total_items)
orders = distribute_items(skus, number_of_orders)
visualize_sku_distribution(orders)







