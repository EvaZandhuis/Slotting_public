import os
import csv
import re
from collections import defaultdict

# Directory path
folder_path = r"C:\Users\pnl0j327\PycharmProjects\pythonProject1\Gall_step12_S100_d1000_T120"

# CSV header with corrected titles
header = ["Demandname", "weights [t, t', t'']", "distribution (pods/item) [t, t', t'']",
          "distribution (items/pod) [t, t', t'']", "distribution (pods/SKU) [t, t', t'']", "seed"]

# Dictionary to hold data grouped by demand name
data_by_demandname = defaultdict(list)

# Function to round values to 4 decimal places
def round_value(value):
    return round(value, 4)

# Function to extract information from a file
def extract_info(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filename = os.path.basename(file_path)
    # Correct pattern to match the filename format
    match = re.match(r"(\w+)_\[\[(.*?)\]\]_(\d+)_\[(.*?)\]", filename)
    if match:
        demandname, weights, seed, dist_pods_item = match.groups()
    else:
        print(f"Filename {filename} does not match the expected pattern.")
        return None

    # Initialize variables to store values for t, t', t''
    values = []

    for i in range(11, 14):  # Updated line range
        line = lines[i].strip()
        try:
            category_match = re.search(r"Category ([\d\.]+)", line)
            if category_match:
                category = float(category_match.group(1))
            else:
                print(f"Category not found in line: {line}")
                return None

            avg_distribution_match = re.search(r"Average Distribution = ([\d\.]+)", line)
            if avg_distribution_match:
                avg_distribution = float(avg_distribution_match.group(1))
            else:
                print(f"Average Distribution not found in line: {line}")
                return None

            boundaries_match = re.search(r"boundaries: \[([\d\.]+), ([\d\.]+)\]", line)
            if boundaries_match:
                boundaries = [float(x) for x in boundaries_match.groups()]
            else:
                print(f"Boundaries not found in line: {line}")
                return None

            items_per_pod_match = re.search(r"(\d+\.?\d*) items per pod", line)
            if items_per_pod_match:
                items_per_pod = float(items_per_pod_match.group(1))
            else:
                print(f"Items per pod not found in line: {line}")
                return None

            pods_per_item_match = re.search(r"so (\d+\.?\d*) pods per 1 item", line)
            if pods_per_item_match:
                pods_per_item = float(pods_per_item_match.group(1))
            else:
                print(f"Pods per item not found in line: {line}")
                return None

            values.append((category, avg_distribution, boundaries, items_per_pod, pods_per_item))
        except Exception as e:
            print(f"Error processing line {i + 1} in file {filename}: {e}")
            return None

    if len(values) != 3:
        print(f"Expected 3 lines of values but got {len(values)} in file {filename}")
        return None

    # Sort values to get t, t', t''
    values.sort(key=lambda x: x[0], reverse=True)
    avg_sorted = [values[0][1], values[1][1], values[2][1]]
    pods_item_sorted = [values[0][4], values[1][4], values[2][4]]
    item_pods_sorted = [values[0][3], values[1][3], values[2][3]]

    rounded_weights = [round_value(float(x.strip())) for x in
                       weights.split(',')]  # Assuming weights is a string like "0.1, 0.2, 0.3"

    return [demandname, f"{[round_value(x) for x in rounded_weights]}", f"{[round_value(x) for x in pods_item_sorted]}",
            f"{[round_value(x) for x in item_pods_sorted]}", f"{[round_value(x) for x in avg_sorted]}", seed]

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        info = extract_info(file_path)
        if info:
            data_by_demandname[info[0]].append(info)

# Write to separate CSV files for each demand name
for demandname, data in data_by_demandname.items():
    # Reset min and max values for each demand
    min_pods_over_item_t = float('inf')
    max_pods_over_item_t = 0
    min_pods_over_item_t2 = float('inf')
    max_pods_over_item_t2 = 0
    min_pods_over_item_t3 = float('inf')
    max_pods_over_item_t3 = 0
    min_items_over_pod_t = float('inf')
    max_items_over_pod_t = 0
    min_items_over_pod_t2 = float('inf')
    max_items_over_pod_t2 = 0
    min_items_over_pod_t3 = float('inf')
    max_items_over_pod_t3 = 0
    min_pods_over_sku_t = float('inf')
    max_pods_over_sku_t = 0
    min_pods_over_sku_t2 = float('inf')
    max_pods_over_sku_t2 = 0
    min_pods_over_sku_t3 = float('inf')
    max_pods_over_sku_t3 = 0

    # Calculate the min and max for this demand
    for info in data:
        pods_item_sorted = eval(info[2])  # Convert string back to list
        item_pods_sorted = eval(info[3])  # Convert string back to list
        avg_sorted = eval(info[4])        # Convert string back to list

        min_pods_over_sku_t = min(min_pods_over_sku_t, avg_sorted[0])
        max_pods_over_sku_t = max(max_pods_over_sku_t, avg_sorted[0])
        min_pods_over_sku_t2 = min(min_pods_over_sku_t2, avg_sorted[1])
        max_pods_over_sku_t2 = max(max_pods_over_sku_t2, avg_sorted[1])
        min_pods_over_sku_t3 = min(min_pods_over_sku_t3, avg_sorted[2])
        max_pods_over_sku_t3 = max(max_pods_over_sku_t3, avg_sorted[2])

        min_pods_over_item_t = min(min_pods_over_item_t, pods_item_sorted[0])
        max_pods_over_item_t = max(max_pods_over_item_t, pods_item_sorted[0])
        min_pods_over_item_t2 = min(min_pods_over_item_t2, pods_item_sorted[1])
        max_pods_over_item_t2 = max(max_pods_over_item_t2, pods_item_sorted[1])
        min_pods_over_item_t3 = min(min_pods_over_item_t3, pods_item_sorted[2])
        max_pods_over_item_t3 = max(max_pods_over_item_t3, pods_item_sorted[2])
        min_items_over_pod_t = min(min_items_over_pod_t, item_pods_sorted[0])
        max_items_over_pod_t = max(max_items_over_pod_t, item_pods_sorted[0])
        min_items_over_pod_t2 = min(min_items_over_pod_t2, item_pods_sorted[1])
        max_items_over_pod_t2 = max(max_items_over_pod_t2, item_pods_sorted[1])
        min_items_over_pod_t3 = min(min_items_over_pod_t3, item_pods_sorted[2])
        max_items_over_pod_t3 = max(max_items_over_pod_t3, item_pods_sorted[2])

    # Write to CSV file for this demand
    output_file = os.path.join(folder_path, f"demand_{demandname}_overview.csv")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header + ["range (pods/item) [t, t', t'']", "range (items/pod) [t, t', t'']",
                                  "range (pods/SKU) [t, t', t'']"])

        # Calculate the ranges for each demand
        range_pods_item = f"[{round_value(min_pods_over_item_t)}, {round_value(max_pods_over_item_t)}], [{round_value(min_pods_over_item_t2)}, {round_value(max_pods_over_item_t2)}], [{round_value(min_pods_over_item_t3)}, {round_value(max_pods_over_item_t3)}]"
        range_items_pod = f"[{round_value(min_items_over_pod_t)}, {round_value(max_items_over_pod_t)}], [{round_value(min_items_over_pod_t2)}, {round_value(max_items_over_pod_t2)}], [{round_value(min_items_over_pod_t3)}, {round_value(max_items_over_pod_t3)}]"
        range_pods_sku = f"[{round_value(min_pods_over_sku_t)}, {round_value(max_pods_over_sku_t)}], [{round_value(min_pods_over_sku_t2)}, {round_value(max_pods_over_sku_t2)}], [{round_value(min_pods_over_sku_t3)}, {round_value(max_pods_over_sku_t3)}]"


# Write each row with the calculated ranges
        for index, row in enumerate(data):
            if index == 0:
                writer.writerow(row + [range_pods_item, range_items_pod, range_pods_sku])
            else:
                writer.writerow(row)

print("Overview files created for each demand name.")
