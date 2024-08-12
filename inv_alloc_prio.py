from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import csv
import random




def large_parameters():
    # ---- Sets ----
    I = range(100)
    J = range(60)

    # ---- Parameters ----
    D = []
    D.extend([58] * 10)
    D.extend([10] * 10)
    D.extend([4] * 80)

    random.shuffle(D)

    S = I
    P = len(J)
    B = 20
    V = 4

    penalty = 0.001

    obj1_weight = 1
    obj2_weight = 0
    obj1_priority = 10
    obj2_priority = 0
    objr_weight = [1, 7.513, 0.052]
    objr_priority = [1, 1, 1]
    weights = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, objr_weight, objr_priority]

    return I, J, D, S, P, B, V, penalty, weights
def test_parameters():
    # ---- Sets ----
    I = range(10)
    J = range(21)

    # ---- Parameters ----
    D = [31, 31, 5, 5, 5, 5, 2, 2, 2, 2]

    S = I
    P = len(J)
    B = 6
    V = 2

    penalty = 0.01

    obj1_weight = 1
    obj2_weight = 0
    obj1_priority = 1
    obj2_priority = 0
    objr_weight = [1, 1, 1]
    objr_priority = [1, 1, 1]
    weights = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, objr_weight, objr_priority]

    return I, J, D, S, P, B, V, penalty, weights
def pick_parameters(size):
    function_name = f"{size}_parameters"
    return getattr(sys.modules[__name__], function_name)()




def run_optimization(scenario_type, scenario_size, simulation_duration, name, filepath):
    model = Model('Inventory allocation')
    start_time = time.time()

    if scenario_size == 'test' or scenario_size == 'large':
        I, J, D, S, P, B, V, penalty, weights = pick_parameters(scenario_size)
        scenario_name = f"{scenario_size} {scenario_type}, V={V},time={simulation_duration}"
    else:
        I, J, D, S, P, B, V, penalty, weights = scenario_size
        scenario_name = name

    obj1_weight = weights[0]
    obj2_weight = weights[1]
    obj1_priority = weights[2]
    obj2_priority = weights[3]
    objr_weight = weights[4]
    objr_priority = weights[5]

    w = weights[4]



    categories = {}
    for i in I:
        category_value = D[i]
        if category_value not in categories:
            categories[category_value] = []
        categories[category_value].append(i)
    sorted_categories = sorted(categories.items(), key=lambda item: item[0], reverse=True)

    # ---- Decision variables ----
    x = {}
    for i in I:
        for j in J:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name='x[' + str(i) + ',' + str(j) + ']')

    y = {}
    for i in I:
        for j in J:
            y[i, j] = model.addVar(lb=0, vtype=GRB.INTEGER, name='y[' + str(i) + ',' + str(j) + ']')

    # ---- Balancing variables ----

    y_bar = {}
    for i in I:
        y_bar[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y_bar[' + str(i) + ']')

    balance_term = {}
    for i in I:
        balance_term[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='balance_term[' + str(i) + ']')

    dist1 = {}
    for i in I:
        for j in J:
            dist1[i, j] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                       name='dist1[' + str(i) + ',' + str(j) + ']')

    abs1 = {}
    for i in I:
        for j in J:
            abs1[i, j] = model.addVar(vtype=GRB.CONTINUOUS, name='abs1[' + str(i) + ',' + str(j) + ']')

    # ---- Priority variables ----

    max_distribution = {}
    for t, i_in_t in categories.items():
        max_distribution[t] = model.addVar(lb=0, vtype=GRB.INTEGER, name='max_distribution[' + str(t) + ']')

    min_distribution = {}
    for t, i_in_t in categories.items():
        min_distribution[t] = model.addVar(lb=0, vtype=GRB.INTEGER, name='min_distribution[' + str(t) + ']')

    average_distribution = {}
    for t, i_in_t in categories.items():
        average_distribution[t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='average_distribution[' + str(t) + ']')

    ratio = {}
    for t, i_in_t in categories.items():
        ratio[t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='ratio[' + str(t) + ']')

    pods_occ = {}
    for t, i_in_t in categories.items():
        pods_occ[t] = model.addVar(lb=0, vtype=GRB.INTEGER, name='pods_occ[' + str(t) + ']')

    skus_in_T = {}
    for t, i_in_t in categories.items():
        skus_in_T[t] = model.addVar(lb=0, vtype=GRB.INTEGER, name='skus_in_T[' + str(t) + ']')

    eq = {}
    for index in [0, 1, 2]:
        eq[index] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='eq[' + str(index) + ']')

    diff = {}
    for index in [0, 1, 2]:
        diff[index] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='diff[' + str(index) + ']')

    abs_diff = {}
    for index in [0, 1, 2]:
        abs_diff[index] = model.addVar(vtype=GRB.CONTINUOUS, name='abs_diff[' + str(index) + ']]')

    model.update()

    # ---- Objective Function ----
    obj1 = model.addVar(vtype=GRB.CONTINUOUS, name="obj1")
    model.addConstr(obj1 == quicksum(x[i, j] for (i, j) in x))
    #model.addConstr(obj1 == P * V)

    obj2 = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj2")
    model.addConstr(obj2 == (-penalty) * quicksum(balance_term[i] for i in I))

    #r_objectives = {}
    #for t, i_in_t in categories.items():
    #   r_objectives[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_objective_{t}")
    #   model.addConstr(r_objectives[t] == average_distribution[t], name=f"r_objective_constraint_{t}")

    #r_objectives = {}
    #for t, i_in_t in categories.items():
    #    r_objectives[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_objective_{t}")
    #    model.addConstr(r_objectives[t] == quicksum(x[i, j] for i in i_in_t for j in J), name=f"r_objective_constraint_{t}")

    #r_objectives = {}
    #for t, i_in_t in categories.items():
    #    r_objectives[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_objective_{t}")
    #    model.addConstr(r_objectives[t] == ratio[t], name=f"r_objective_constraint_{t}")

    r_objectives = {}
    for index in [0, 1, 2]:
        r_objectives[index] = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_objective_{index}")
        model.addConstr(r_objectives[index] == abs_diff[index], name=f"r_objective_constraint_{index}")


    #r_objectives = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_objective_{index}")
    #model.addConstr(r_objectives == abs_diff[0] + abs_diff[1], name=f"r_objective_constraint_{index}")

    model.update()

    # ---- scenario's ----
    if scenario_type == 'base':
        #model.setObjective(obj1 + obj2, sense=GRB.MAXIMIZE)
        model.setObjective(obj1, sense=GRB.MAXIMIZE)

    if scenario_type == 'prio':
        model.modelSense = GRB.MAXIMIZE
        model.setObjectiveN(obj1, 0, priority=len(categories.keys()) + 1)
        index_value = 1
        priority_value = len(categories.keys())
        for t in sorted(categories.keys(), reverse=True):
            model.setObjectiveN(r_objectives[t], index_value, priority_value)
            index_value += 1
            priority_value -= 1
        #model.setObjectiveN(obj2, index_value, priority_value)

    if scenario_type == 'weighted':
        model.modelSense = GRB.MAXIMIZE
        model.setObjectiveN(obj1, index=0, priority=obj1_priority, weight=obj1_weight)
        model.setObjectiveN(r_objectives[0], index=1, priority=objr_priority[0], weight=-1)
        model.setObjectiveN(r_objectives[1], index=2, priority=objr_priority[1], weight=-1)
        #model.setObjectiveN(r_objectives[2], index=3, priority=objr_priority[2], weight=-1)
        #model.setObjectiveN(obj2, index=4, priority=obj2_priority, weight=obj2_weight)






    model.update()

    # ---- Constraints ----
    con3 = {}
    for j in J:
        con3[j] = model.addConstr(quicksum(y[i, j] for i in I) <= B)

    con4 = {}
    for j in J:
        con4[j] = model.addConstr(quicksum(x[i, j] for i in I) <= V)

    con5 = {}
    for i in I:
        con5[i] = model.addConstr(quicksum(y[i, j] for j in J) == D[i])

    con6 = {}
    for i in I:
        for j in J:
            con6[i, j] = model.addConstr(y[i, j] >= x[i, j])

    con7 = {}
    for i in I:
        for j in J:
            con7[i, j] = model.addConstr(y[i, j] <= B * x[i, j])

    # ---- Balancing constraints ----

    con8 = {}
    for i in I:
        con8[i] = model.addConstr(D[i] == y_bar[i] * (quicksum(x[i, j] for j in J)))

    con9 = {}
    for i in I:
        for j in J:
            con9[i, j] = model.addConstr(dist1[i, j] == y[i, j] - y_bar[i])

    conAbs = {}
    for i in I:
        for j in J:
            conAbs[i, j] = model.addGenConstrAbs(abs1[i, j], dist1[i, j])

    con10 = {}
    for i in I:
        con10[i] = model.addConstr(balance_term[i] == quicksum(abs1[i, j] * x[i, j] for j in J))

    # ---- Priority constraints ----

    con11 = {}
    for t, i_in_t in categories.items():
        con11[t] = model.addConstr(pods_occ[t] == quicksum(x[i, j] for i in i_in_t for j in J))

    con12 = {}
    for t, i_in_t in categories.items():
        con12[t] = model.addConstr(skus_in_T[t] == len(i_in_t))

    con13 = {}
    for t, i_in_t in categories.items():
        con13[t] = model.addConstr(pods_occ[t] == average_distribution[t] * skus_in_T[t])

    con14 = {}
    for t, i_in_t in categories.items():
        con14[t] = model.addConstr(min_distribution[t] == max(1, math.ceil(t / B)))

    con15 = {}
    for t, i_in_t in categories.items():
        con15[t] = model.addConstr(max_distribution[t] == min(t, len(J)))
    epsilon = 0.00001
    #con16 = {}
    #for t, i_in_t in categories.items():
    #    con16[t] = model.addConstr((average_distribution[t] - min_distribution[t]) * 100 == ratio[t] * (
    #                max_distribution[t] - min_distribution[t] + epsilon))

    #con16 = {}
    #for t, i_in_t in categories.items():
    #    con16[t] = model.addConstr(average_distribution[t] == ratio[t] * D[t])

    con16 = {}
    for index, (t, i_in_t) in enumerate(sorted_categories):
        con16[t] = model.addConstr(eq[index] * skus_in_T[t] * t == w[index] * quicksum(x[i, j] for i in i_in_t for j in J))

    con_diff = {}
    con_diff[0] = model.addConstr(diff[0] == eq[0] - eq[1])
    con_diff[1] = model.addConstr(diff[1] == eq[0] - eq[2])
    con_diff[2] = model.addConstr(diff[2] == eq[1] - eq[2])

    con_abs_diff = {}
    con_abs_diff[0] = model.addGenConstrAbs(abs_diff[0], diff[0])
    con_abs_diff[1] = model.addGenConstrAbs(abs_diff[1], diff[1])
    con_abs_diff[2] = model.addGenConstrAbs(abs_diff[2], diff[2])

    model.update()

    # ---- Visualise ----
    def print_pods():
        for j in J:
            print(f"Pod {j}:")

            # Get the SKUs and quantities assigned to the current pod
            assigned_skus = [i for i in I if y[i, j].x > 0.5]
            quantities = [y[i, j].x for i in assigned_skus]

            # Print the SKUs and quantities
            for i, quantity in zip(assigned_skus, quantities):
                print(f"  SKU {i}: {quantity}")

            print()
    def print_pods_to_csv(filename, filepath):
        if filepath is None:
            path = filename
        else:
            path = os.path.join(filepath, filename)
        with open(path, "w", newline='') as csvfile:
            fieldnames = ['Pod Index', 'SKU Index', 'Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for j in J:
                pod_index = j
                assigned_skus = [i for i in I if y[i, j].x > 0.5]
                quantities = [int(y[i, j].x) for i in assigned_skus]

                for i, quantity in zip(assigned_skus, quantities):
                    for _ in range(quantity):  # Repeat for each count
                        writer.writerow({'Pod Index': pod_index, 'SKU Index': i, 'Count': 1})
    def check_problem(print_to_console=True):  # ---- What is max x[i,j] with this number of pods
        count = 0
        for i in I:
            if D[i] >= len(J):
                count += len(J)
            else:
                count += D[i]
        if print_to_console:
            print('max spread with pods as constraints =', count)
        return str(count)
    def visualize_results(filepath, print_to_console=True):
        # Assuming model is already optimized

        # Create a single plot with stacked bars for each pod
        if len(J) > 50:
            plt.figure(figsize=(20, 10))
        else:
            plt.figure()
        plt.title(f"Pods - SKU Assignment (sim_time{simulation_duration}, obj{scenario_type})")
        plt.xlabel("Pod [index]")
        plt.ylabel("Quantity [items]")

        # Define colors inside the function
        original_colors = plt.cm.rainbow(np.linspace(0, 1, len(I)))

        sorted_keys = sorted(categories.keys(), reverse=True)
        sorted_indices = []
        for key in sorted_keys:
            sorted_indices.extend(categories[key])
        colors = np.zeros_like(original_colors)
        for new_index, old_index in enumerate(sorted_indices):
            colors[old_index] = original_colors[new_index]

        # Initialize legend_labels outside the loop
        legend_labels = []

        # Iterate over each pod (j)
        for j in J:
            # Initialize the bottom position for each SKU in the pod
            bottom_positions = np.zeros(len(I))

            # Iterate over each SKU (i) in the pod
            for i in I:
                quantity = y[i, j].x

                # Update the bottom position for the next SKU
                if i + 1 < len(I):
                    bottom_positions[i + 1] += quantity + bottom_positions[i]

                # Plot a bar for the current SKU with the assigned quantity
                #plt.bar(j, quantity, width=0.8, bottom=bottom_positions[i], color=colors[i], label=f"SKU{i}")

                # Add SKU to legend_labels only if it hasn't been added before
                if i not in legend_labels:
                    plt.bar(j, quantity, width=0.8, bottom=bottom_positions[i], color=colors[i], label=f"SKU{i} (x{D[i]})")
                    legend_labels.append(i)

                else:
                    plt.bar(j, quantity, width=0.8, bottom=bottom_positions[i], color=colors[i])

        # Create the legend using legend_labels
        if len(J) > 50:
            # Initialize legend handles and labels for categories
            category_legend_handles = []
            category_legend_labels = []

            for category_value, skus in categories.items():
                first_sku_index = skus[0]
                last_sku_index = skus[-1]

                # Get colors for the first and last SKUs in the category
                first_sku_color = colors[first_sku_index]
                last_sku_color = colors[last_sku_index]

                # Plot dummy bars for legend for the first SKU
                first_dummy_handle = plt.bar(0, 0, color=first_sku_color,
                                             label=f"SKU{first_sku_index} - {category_value}")
                category_legend_handles.append(first_dummy_handle)
                category_legend_labels.append(f"SKU{first_sku_index} - {category_value}")

                # Plot dummy bars for legend for the last SKU
                last_dummy_handle = plt.bar(0, 0, color=last_sku_color, label=f"SKU{last_sku_index} - {category_value}")
                category_legend_handles.append(last_dummy_handle)
                category_legend_labels.append(f"SKU{last_sku_index} - {category_value}")

                # Add an empty line as a separator between categories
                empty_line_handle = plt.Line2D([0], [0], linestyle='None', label='', alpha=0)  # Placeholder handle
                category_legend_handles.append(empty_line_handle)
                category_legend_labels.append('')  # Empty label

            plt.legend(handles=category_legend_handles, labels=category_legend_labels, bbox_to_anchor=(1.05, 1),
                       loc='upper left', fontsize=8)
            plt.xticks(np.arange(len(J)), [str(j) for j in J], fontsize=5, rotation=90)
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            #plt.legend([f"SKU{i}" for i in legend_labels], bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(np.arange(len(J)), [str(j) for j in J])
            plt.tight_layout()
        if filepath is None:
            plt.savefig(scenario_name + ".png")
        else:
            plt.savefig(os.path.join(filepath, scenario_name + ".png"))
        plt.close()
        if print_to_console:
            plt.show()
    def print_prio(print_to_console=True):
        categories = {}
        for i in I:
            category_value = D[i]
            if category_value not in categories:
                categories[category_value] = []
            categories[category_value].append(i)

        result_str = ""



        # Calculate average distribution per category
        for category_value, skus_in_category in categories.items():
            total_pods_occupied = sum(x[i, j].x for i in skus_in_category for j in J)
            total_skus_in_category = len(skus_in_category)
            average_distribution = total_pods_occupied / total_skus_in_category
            min_distribution = max(1, math.ceil(category_value / B))
            max_distribution = min(category_value, len(J))
            if max_distribution - min_distribution == 0:
                ratio = 1
            else:
                ratio = (average_distribution - min_distribution) / (max_distribution - min_distribution + epsilon)
            sum_distr = (average_distribution - min_distribution) * total_skus_in_category
            items_per_pod = category_value / average_distribution
            pods_per_item = average_distribution / category_value
            #ratio_spread = total_pods_occupied / sum(x[i, j].x for i in I for j in J)
            result_str += f"Category {category_value} (x{total_skus_in_category}): Average Distribution = {average_distribution}, " \
                          f"With boundaries: [{min_distribution}, {max_distribution}], " \
                          f"With {items_per_pod} items per pod, so {pods_per_item} pods per 1 item."\
                          f"With sum_distr: {sum_distr}" \
                          f"Ratio = {ratio * 100} percent of possible pods occupied\n" \


                        # f"new ratio = {ratio_spread * 100} percent of occupation, "\

        if print_to_console:
            print(result_str)
        return result_str
    def write_results(filepath):
        filename = scenario_name + ".txt"
        if filepath is None:
            path = filename
        else:
            path = os.path.join(filepath, filename)
        with open(path, "w") as file:
            file.write(scenario_name + "\n")
            file.write(str(D))
            file.write("\n")
            file.write(f'weights: w1={obj1_weight}, wr={objr_weight}, w2={obj2_weight}\n')
            file.write(f'priorities: p1={obj1_priority}, pr={objr_priority}, p2={obj2_priority} \n')
            file.write("\n")
            file.write("objective value = " + str(model.objVal) + "\n")
            file.write("max spread with pods as constraints = " + check_problem(print_to_console=False) + "\n")
            file.write("\n")
            file.write("penalty term = -" + str(penalty) + " balancing term = " + str(
                sum(balance_term[i].x for i in I)) + "\n")
            file.write("penalty = -" + str(penalty * sum(balance_term[i].x for i in I)) + "\n")
            file.write("\n")
            file.write(print_prio(print_to_console=False) + "\n")
            visualize_results(filepath, print_to_console=False)  # save image
            if w[0] == 0:
                file.write(f'distribution t(high): {round(eq[0].x, 3)} (pods per item)\n')
            else:
                file.write(f'distribution t(high): {round(eq[0].x/w[0], 3)} (pods per item)\n')
            if w[1] == 0:
                file.write(f'distribution t(mid) : {round(eq[1].x, 3)} (pods per item)\n')
            else:
                file.write(f'distribution t(mid) : {round(eq[1].x / w[1], 3)} (pods per item)\n')
            if w[2] == 0:
                file.write(f'distribution t(low) : {round(eq[2].x, 3)} (pods per item)\n')
            else:
                file.write(f'distribution t(low) : {round(eq[2].x / w[2], 3)} (pods per item)\n')
            file.write("\n")
            file.write(f'eq1       =       eq2       =       eq3\n')
            file.write(f'd1  *  wr1  =  d2  *  rw2  =  d3  *  wr3\n')
            w0, w1, w2 = [1 if x == 0 else x for x in w[:3]]
            file.write(f'{round(eq[0].x / w0, 3)} * {w[0]}  =  {round(eq[1].x / w1, 3)} * {w[1]}  =  {round(eq[2].x / w2, 3)} * {w[2]}\n')
            file.write("\n")
            file.write(f'With eq1={round(eq[0].x, 3)} \n')
            file.write(f'With eq2={round(eq[1].x, 3)} \n')
            file.write(f'With eq3={round(eq[2].x, 3)} \n')
            file.write(f'diff1={round(abs_diff[0].x, 3)}=eq1-eq2 \n')
            file.write(f'diff2={round(abs_diff[1].x, 3)}=eq1-eq3 \n')
            file.write(f'diff3={round(abs_diff[2].x, 3)}=eq2-eq3 \n')
            file.write("\n")
            t_low, t_mid, t_high = sorted(categories.keys())[:3]
            min_w2 = (max_distribution[t_high].x / t_high) / (min_distribution[t_mid].x / t_mid)
            max_w2 = (min_distribution[t_high].x / t_high) / (max_distribution[t_mid].x / t_mid)
            min_w3 = (max_distribution[t_high].x / t_high) / (min_distribution[t_low].x / t_low)
            max_w3 = (min_distribution[t_high].x / t_high) / (max_distribution[t_low].x / t_low)
            file.write(f'range(w2)=({round(min_w2, 3)}, {round(max_w2, 3)})\n')
            file.write(f'range(w3)=({round(min_w3, 3)}, {round(max_w3, 3)})\n')
            file.write("\n")
            file.write("set simulation duration = " + str(simulation_duration) + "\n")
            file.write('actual runtime = ' + str(end_time - start_time) + 'sec' + "\n")

    model.setParam('OutputFlag', True)  # silencing gurobi output or not (True=output, false=no output)
    #model.setParam('MIPFocus', 2)
    #model.setParam('MIPGap', 0)
    #model.setParam('Method', 1) # find the optimal solution
    model.params.LogFile = 'Inventory allocation.log'
    if simulation_duration is not None:
        model.Params.timeLimit = simulation_duration
    # model.computeIIS()
    # model.write("output.lp") # print the model in .lp format file
    model.optimize()

    if w[0] == 0:
        z_1 = round(eq[0].x, 3)
    else:
        z_1 = round(eq[0].x / w[0], 3)
    if w[1] == 0:
        z_2 = round(eq[1].x, 3)
    else:
        z_2 = round(eq[1].x / w[1], 3)
    if w[2] == 0:
        z_3 = round(eq[2].x, 3)
    else:
        z_3 = round(eq[2].x / w[2], 3)
    scenario_name = f"{scenario_name}_[{z_1}, {z_2}, {z_3}]"

    end_time = time.time()
    #print(model.objVal)
    # check_problem()
    # print_pods()
    visualize_results(filepath, False)
    # print_prio()
    print('runtime =', end_time - start_time, 'sec')

    print_pods_to_csv(scenario_name + ".csv", filepath)
    write_results(filepath)
    model = None


# args are: (scenario_type, scenario_size, simulation_duration) => ('base'/'prio, 'test'/'large', 10/sec/None)
#run_optimization('weighted', 'large', 60, None, None)
