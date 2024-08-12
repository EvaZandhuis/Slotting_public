import os
import re
import csv

def extract_avg_distribution(lines):
    distributions = []
    for line in lines:
        category_match = re.search(r'Category\s+(\d+(\.\d+)?)', line)
        avg_distr_match = re.search(r'Average Distribution\s*=\s*(\d+(\.\d+)?)', line)

        if category_match and avg_distr_match:
            category = float(category_match.group(1))
            avg_distr = float(avg_distr_match.group(1))
            distributions.append((category, avg_distr))

    # Sort by category in descending order
    distributions.sort(reverse=True, key=lambda x: x[0])

    # Create a string of average distributions
    avg_distr_str = ', '.join([f'avg={d[1]}' for d in distributions])
    return avg_distr_str

def read_files_with_prefix(directory, prefix):
    output_filename = os.path.join(directory, f"weight_config_{prefix}.csv")
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Name', 'Weights', 'Distribution', 'Average Distribution'])

        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as file:
                    lines = file.readlines()
                    if len(lines) < 18:
                        print(f"File {filename} does not have enough lines to extract the required information.")
                        continue

                    # Extract the first line and split it
                    first_line = lines[0].strip()
                    parts = first_line.split('_')
                    name = f"{parts[0]}_{parts[-1]}"

                    # Extract weights from the first line using regular expressions
                    weights_str = re.search(r'\[\[.*\]\]', first_line).group(0)

                    # Extract distribution values from lines 16, 17, and 18
                    dist_line_16 = lines[15].strip().replace(' (pods per item)', '')
                    dist_line_17 = lines[16].strip().replace(' (pods per item)', '')
                    dist_line_18 = lines[17].strip().replace(' (pods per item)', '')

                    # Format distribution values
                    distribution = f"{dist_line_16.split(' ', 1)[1]}, {dist_line_17.split(' ', 1)[1]}, {dist_line_18.split(' ', 1)[1]}"

                    # Extract and format average distributions from lines 12, 13, and 14
                    avg_distribution = extract_avg_distribution(lines[11:14])

                    # Write the formatted output to the CSV file
                    csvwriter.writerow([name, weights_str, distribution, avg_distribution])
                    print(f"{name:<10} {weights_str:<40} {distribution:<50} {avg_distribution}")


# Example usage
directory_path = r'C:\Users\pnl0j327\PycharmProjects\pythonProject1\Gall_S100_d1000_T120'
prefix = 'Dp'  # Replace with the desired prefix
read_files_with_prefix(directory_path, prefix)

