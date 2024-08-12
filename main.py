from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import csv
from inv_alloc_prio import run_optimization
from three_s import demand_sku_distribution
import os
import pandas as pd
import ast
import random






# demand parameters
S = 100
demand = 1000
#S = 10
#demand = 100
s_values = [0.222, 0.139, 0.065]
colors = ['g', 'orange', 'r']
class_i1l = 0.1
class_i1r = 0.2
class_i2l = 0.2
class_i2r = 0.6
I = range(S)
#J = range(21)
J = range(60)
P = len(J)
# saved demand list
csv_file = f'S{S}_demand{demand}_svalues{s_values}_classes{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}.csv'
#csv_file = f'S{S}_demand{demand}.csv'
demand_list_names = ['Da', 'Db', 'Dc', 'Dd', 'De', 'Df']
# slotting parameters
#B = 6
#V = 3
B = 20
V = 4
penalty = 0.001
seeds = 3
simulation_duration = 120
# w
wd_values = [1, 0, 0]
step_values = [0, 0]
# wp
obj1_weight = 1
obj2_weight = 0
obj1_priority = 10
obj2_priority = 0
objr_weight = [1, 1, 1]
objr_priority = [9, 8, 7]
p_weights = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, objr_weight, objr_priority]
# (w1, w2, p1, p2, w3, p3)

# op prio volgorde van t(high) naar t(low):
#prio_high_weights = [1, 0, 10, 0, [1, 1, 1], [9, 7, 5]]
# twee uiterste scenarios min(xij) en max(xij):
min_xij_weights = [-1, 0, 1, 0, [0, 0, 0], [0, 0, 0]]
random_weights = [1, 0, 1, 0, [0, 0, 0], [0, 0, 0]]

range_w2_Da = [0.052, 10]
range_w3_Da = [0.052, 1.039]
steps_w2_Da = 6
steps_w3_Da = 4
range_w2_Db =[0.059, 1.295]
range_w3_Db =[0.059, 0.794]
steps_w2_Db = 6
steps_w3_Db = 4
range_w2_Dc =[0.058, 6.087]
range_w3_Dc =[0.058, 0.7]
steps_w2_Dc = 6
steps_w3_Dc = 4
range_w2_Dd =[0.05, 0.79]
range_w3_Dd =[0.05, 0.54]
steps_w2_Dd = 6
steps_w3_Dd = 4
range_w2_De =[0.058, 1]
range_w3_De =[0.058, 0.058]
steps_w2_De = 5
steps_w3_De = 1
range_w2_Df =[0.1, 0.38]
range_w3_Df =[0.068, 0.068]
#range_w2_Df =[2, 0.068]
#range_w3_Df =[1, 0.068]
steps_w2_Df = 7
steps_w3_Df = 1


def generate_specific_weight_configurations(demand_name, add_min, add_random):
    # Fixed priorities
    obj1_weight = 1
    obj1_priority = 10
    obj2_weight = 0
    obj2_priority = 0
    objr_priority = [1, 1, 1]

    if demand_name == 'Da':
        range_w2 = range_w2_Da
        range_w3 = range_w3_Da
        w2_steps = steps_w2_Da
        w3_steps = steps_w3_Da
    elif demand_name == 'Db':
        range_w2 = range_w2_Db
        range_w3 = range_w3_Db
        w2_steps = steps_w2_Db
        w3_steps = steps_w3_Db
    elif demand_name == 'Dc':
        range_w2 = range_w2_Dc
        range_w3 = range_w3_Dc
        w2_steps = steps_w2_Dc
        w3_steps = steps_w3_Dc
    elif demand_name == 'Dd':
        range_w2 = range_w2_Dd
        range_w3 = range_w3_Dd
        w2_steps = steps_w2_Dd
        w3_steps = steps_w3_Dd
    elif demand_name == 'De':
        range_w2 = range_w2_De
        range_w3 = range_w3_De
        w2_steps = steps_w2_De
        w3_steps = steps_w3_De
    elif demand_name == 'Df':
        range_w2 = range_w2_Df
        range_w3 = range_w3_Df
        w2_steps = steps_w2_Df
        w3_steps = steps_w3_Df
    elif demand_name == 'Dr':
        range_w2 = [0.01, 1.6]
        range_w3 = [1, 1]
        w2_steps = 12
        w3_steps = 1
    elif demand_name == 'Dp':
        range_w2 = [0.01, 1]
        range_w3 = [1, 1]
        w2_steps = 12
        w3_steps = 1
    else:
        raise ValueError("Invalid demand name")

    w2_values = np.linspace(range_w2[0], range_w2[1], w2_steps)
    w3_values = np.linspace(range_w3[0], range_w3[1], w3_steps)

    # List to hold all weight configurations
    all_weight_configurations = []
    weight_list_names = []

    weights111_exists = 0

    for w2_value in w2_values:
        for w3_value in w3_values:
            wr_weights = [1, w2_value, w3_value]
            if abs(1 - round(w2_value, 2)) <= 0.04 and abs(1 - round(w3_value, 2)) <= 0.04:
                weights111_exists = 1
            config = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, wr_weights, objr_priority]
            all_weight_configurations.append(config)
            weight_list_names.append(f'[{wr_weights}]')

    if weights111_exists == 0:
        wr_weights = [1, 1, 1]
        config = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, wr_weights, objr_priority]
        all_weight_configurations.append(config)
        weight_list_names.append(f'[{wr_weights}]')


    if add_min == 'y':
        all_weight_configurations.append(min_xij_weights)
        weight_list_names.append('wmin')

    if add_random == 'y':
        all_weight_configurations.append(random_weights)
        weight_list_names.append('wr')
    for index, name in enumerate(weight_list_names):
        print(f'{index}. {name}')

    return all_weight_configurations, weight_list_names
def generate_general_weight_configurations():
    # Fixed priorities
    obj1_weight = 1
    obj1_priority = 1
    obj2_weight = 0
    obj2_priority = 0
    objr_priority = [1, 1, 1]
    range_w2 = [2, 1, 0.2]
    range_w3 = [2, 1, 0.2]

    # List to hold all weight configurations
    all_weight_configurations = []
    weight_list_names = []

    for w2_value in range_w2:
        for w3_value in range_w3:
            wr_weights = [1, w2_value, w3_value]
            config = [obj1_weight, obj2_weight, obj1_priority, obj2_priority, wr_weights, objr_priority]
            all_weight_configurations.append(config)
            weight_list_names.append(f'[{wr_weights}]')


    return all_weight_configurations, weight_list_names

def read_demand(filename, filepath):
    D1 = []  # s=0.222
    D2 = []  # s=0.139
    D3 = []  # s=0.065

    if filepath is None:
        path = filename
    else:
        path =os.path.join(filepath, filename)
    with open(path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            s_value = float(row['s'])
            demand_value = float(row['Demand'])
            if s_value == 0.222:
                D1.append(demand_value)
            elif s_value == 0.139:
                D2.append(demand_value)
            elif s_value == 0.065:
                D3.append(demand_value)
    #print(D1)
    # print(D2)
    return D1, D2, D3

def manual_adjust_to_integer(demand_list):
    unique_values = set(demand_list)
    counts = {value: demand_list.count(value) for value in unique_values}
    for value, count in counts.items():
        print(f"Value {value} occurs {count} times.")
    rounded_demand_list = [round(value) for value in demand_list]

    while True:
        unique_values = set(rounded_demand_list)
        counts = {value: rounded_demand_list.count(value) for value in unique_values}
        for value, count in counts.items():
            print(f"Value {value} occurs {count} times.")
        print('total demand =', sum(rounded_demand_list), 'and should be', demand)
        old_value = float(input("Enter the value you want to change (or '0' to quit): "))
        if old_value == 0:
            break
        new_value = float(input("Enter the new value: "))
        rounded_demand_list = [new_value if x == old_value else x for x in rounded_demand_list]

    return rounded_demand_list

def save_demand_list(demand_list, demand_name, filepath):
    fieldnames = ['demand_name', 'demand_list']
    if filepath is None:
        path = csv_file
    else:
        path = os.path.join(filepath, csv_file)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()
            csv.DictWriter(file, fieldnames=fieldnames).writerow({'demand_name': demand_name, 'demand_list': demand_list})
    else:
        df = pd.read_csv(path)
        if demand_name not in df['demand_name'].values:
            with open(path, 'a', newline='') as file:
                csv.DictWriter(file, fieldnames=fieldnames).writerow({'demand_name': demand_name, 'demand_list': demand_list})
        else:
            with open(path, 'w', newline='') as file:
                csv.DictWriter(file, fieldnames=fieldnames).writeheader()
                for index, row in df.iterrows():
                    if row['demand_name'] != demand_name:
                        csv.DictWriter(file, fieldnames=fieldnames).writerow({'demand_name': row['demand_name'], 'demand_list': row['demand_list']})
                csv.DictWriter(file, fieldnames=fieldnames).writerow({'demand_name': demand_name, 'demand_list': demand_list})

def load_demand_list(demand_name, filepath):
    if filepath is None:
        path = csv_file
    else:
        path = os.path.join(filepath, csv_file)
    if os.path.exists(path):
        demand_list = []  # Initialize an empty list to store demand lists
        with open(path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = str(row['demand_name'])
                saved_demand_list = row['demand_list']
                if name == demand_name:
                    # Convert the string representation of the list to a Python list
                    demand_sublist = ast.literal_eval(saved_demand_list)
                    # Extend demand_lists with the elements of demand_list
                    demand_list.extend(demand_sublist)
        if demand_list:  # Check if any demand lists were found
            return demand_list
        else:
            return None
    else:
        return None

def manage_demand_list(saved_demand, original_demand, demand_name, filepath):
    if saved_demand is not None:
        print(f"\nPreviously saved manually adjusted demand for {demand_name}:")
        print(saved_demand)
        print('sum is', sum(saved_demand))
        use_saved = input("Do you want to use the previously saved manually adjusted demand? (y/n): ").strip().lower()
        if use_saved == 'y':
            D = saved_demand
        else:
            D = manual_adjust_to_integer(original_demand)
            save_demand_list(D, demand_name, filepath)
    else:
        D = manual_adjust_to_integer(original_demand)
        save_demand_list(D, demand_name, filepath)
    #print(demand_name, D)
    return D

def manage_total_demand(filepath):
    original_Da, original_Dc, original_De = read_demand(f'Demand_values_{class_i1l}-{class_i1r}SKUs{S}_Demand{demand}.csv', filepath)
    original_Db, original_Dd, original_Df = read_demand(f'Demand_values_{class_i2l}-{class_i2r}SKUs{S}_Demand{demand}.csv', filepath)

    saved_Da = load_demand_list('Da', filepath)
    saved_Db = load_demand_list('Db', filepath)
    saved_Dc = load_demand_list('Dc', filepath)
    saved_Dd = load_demand_list('Dd', filepath)
    saved_De = load_demand_list('De', filepath)
    saved_Df = load_demand_list('Df', filepath)

    saved_demands = [saved_Da, saved_Db, saved_Dc, saved_Dd, saved_De, saved_Df]

    if all(saved_demand is not None for saved_demand in saved_demands):
        if all(sum(saved_demand) == demand for saved_demand in saved_demands):
            print(f"\nSaved demand == demand({demand}) for all demand lists")
            quick_use_saved = input("Use saved demand without checking? (y/n): ").strip().lower()
            if quick_use_saved == 'y':
                Da, Db, Dc, Dd, De, Df = saved_demands
                return Da, Db, Dc, Dd, De, Df
            else:
                print("\nProceeding to manage demand lists")
        else:
            print("\nSaved demand does not match given demand for all lists, proceeding to manage demand lists")

    else:
        print("\nNot all saved demand is available (is not None), proceeding to manage demand lists")

    Da = manage_demand_list(saved_Da, original_Da, 'Da', filepath)
    Db = manage_demand_list(saved_Db, original_Db, 'Db', filepath)
    Dc = manage_demand_list(saved_Dc, original_Dc, 'Dc', filepath)
    Dd = manage_demand_list(saved_Dd, original_Dd, 'Dd', filepath)
    De = manage_demand_list(saved_De, original_De, 'De', filepath)
    Df = manage_demand_list(saved_Df, original_Df, 'Df', filepath)

    return Da, Db, Dc, Dd, De, Df

def configure_gall_demand(d, skus):
    D = []

    A_skus = int(skus[0] * S / 100)
    B_skus = int((skus[1] - skus[0]) * S / 100)
    C_skus = int((100 - skus[1]) * S / 100)
    A_demand = (d[0] * demand /  100) / A_skus
    B_demand = (d[1] * demand / 100) / B_skus
    C_demand = ((100 - d[0] - d[1]) * demand /100) /C_skus
    D.extend([A_demand] * A_skus)
    D.extend([B_demand] * B_skus)
    D.extend([C_demand] * C_skus)

    return D

def manage_gall_demand(filepath):
    regular_SKUs = [20, 50]
    regular_Di = [75, 20]
    peak_SKUs = [20, 50]
    peak_Di = [80, 15]

    original_Dr = configure_gall_demand(regular_Di, regular_SKUs)
    original_Dp = configure_gall_demand(peak_Di, peak_SKUs)

    saved_Dr = load_demand_list('Dr', filepath)
    saved_Dp = load_demand_list('Dp', filepath)

    saved_demands = [saved_Dr, saved_Dp]

    if all(saved_demand is not None for saved_demand in saved_demands):
        if all(sum(saved_demand) == demand for saved_demand in saved_demands):
            print(f"\nSaved demand == demand({demand}) for all demand lists")
            quick_use_saved = input("Use saved demand without checking? (y/n): ").strip().lower()
            if quick_use_saved == 'y':
                Dr, Dp = saved_demands
                return Dr, Dp
            else:
                print("\nProceeding to manage demand lists")
        else:
            print("\nSaved demand does not match given demand for all lists, proceeding to manage demand lists")

    else:
        print("\nNot all saved demand is available (is not None), proceeding to manage demand lists")

    Dr = manage_demand_list(saved_Dr, original_Dr, 'Dr', filepath)
    Dp = manage_demand_list(saved_Dp, original_Dp, 'Dp', filepath)

    return Dr, Dp


def manual_adjust_weights(initial_weights):
    obj1_weight = initial_weights[0]
    obj2_weight = initial_weights[1]
    obj1_priority = initial_weights[2]
    obj2_priority = initial_weights[3]
    objr_weight = initial_weights[4]
    objr_priority = initial_weights[5]
    while True:
        print("\nCurrent weights and priorities:")
        print(f"1. w1 (weight obj1) = {obj1_weight}")
        print(f"2. wr1 (weight ratio1) = {objr_weight[0]}")
        print(f"3. wr2 (weight ratio2) = {objr_weight[1]}")
        print(f"4. wr3 (weight ratio3) = {objr_weight[2]}")
        print(f"5. w2 (weight obj2) = {obj2_weight}")
        print(f"6. p1 (priority obj1) = {obj1_priority}")
        print(f"7. pr1 (priority ratio1) = {objr_priority[0]}")
        print(f"8. pr2 (priority ratio2) = {objr_priority[1]}")
        print(f"9. pr3 (priority ratio3) = {objr_priority[2]}")
        print(f"10. p2 (priority obj2) = {obj2_priority}")

        choice = input("\nEnter the line number of the value you want to change (or '0' to finish): ").strip()

        if choice == '0':
            break

        if choice == '1':
            obj1_weight = int(input("Set w1 (weight obj1) (int): ").strip())
        elif choice == '2':
            objr_weight[0] = int(input("Set wr1 (weight ratio1) (int): ").strip())
        elif choice == '3':
            objr_weight[1] = int(input("Set wr2 (weight ratio2) (int): ").strip())
        elif choice == '4':
            objr_weight[2] = int(input("Set wr3 (weight ratio3) (int): ").strip())
        elif choice == '5':
            obj2_weight = int(input("Set w2 (weight obj2) (int): ").strip())
        elif choice == '6':
            obj1_priority = float(input("Set p1 (priority obj1) (float): ").strip())
        elif choice == '7':
            objr_priority[0] = float(input("Set pr1 (priority ratio1) (float): ").strip())
        elif choice == '8':
            objr_priority[1] = float(input("Set pr2 (priority ratio2) (float): ").strip())
        elif choice == '9':
            objr_priority[2] = float(input("Set pr3 (priority ratio3) (float): ").strip())
        elif choice == '10':
            obj2_priority = float(input("Set p2 (priority obj2) (float): ").strip())
        else:
            print("Invalid choice, please try again.")

    print("\nFinal weights and priorities are:")
    print(f'w1={obj1_weight}, wr={objr_weight}, w2={obj2_weight}')
    print(f'p1={obj1_priority}, pr={objr_priority}, p2={obj2_priority}')

    return [obj1_weight, obj2_weight, obj1_priority, obj2_priority, objr_weight, objr_priority]

def manage_slotting_objective():
    #use_weights = input(f"Use weighted scenario type? (y/n): ").strip().lower()
    #if use_weights == 'y':
    #    scenario_type = 'weighted'
    #elif use_weights == 'n':
    #    use_prio = input(f"\nUse priority scenario type? (y/n): ").strip().lower()
    #    if use_prio == 'y':
    #        scenario_type = 'prio'
    #    elif use_prio == 'n':
    #        scenario_type = 'base'
    #print(f"Use scenario {scenario_type}")

    scenario_type = 'weighted'

    if scenario_type == 'weighted':
        specific_weights = input(f"\nUse demand specific weights? (y/n): ").strip().lower()
        add_min = input(f"\nInclude obj: min(x[i,j]) slotting? (y/n): ").strip().lower()
        add_random = input(f"\nInclude (random) obj: max(x[i,j]) slotting? (y/n): ").strip().lower()
        if specific_weights == 'n':
            all_weight_configurations, weight_list_names = generate_general_weight_configurations()

            #all_weight_configurations.append(p_weights)
            #weight_list_names.append('wp')

            if add_min == 'y':
                all_weight_configurations.append(min_xij_weights)
                weight_list_names.append('wmin')

            if add_random == 'y':
                all_weight_configurations.append(random_weights)
                weight_list_names.append('wr')
            #print('Preset weights are:')
            #print(f'w1={p_weights[0]}, wr={p_weights[4]}, w2={p_weights[1]}')
            #print('Preset priorities are:')
            #print(f'p1={p_weights[2]}, pr={p_weights[5]}, p2={p_weights[3]}')
            #use_set_weights = input(f"\nUse weights and priorities from parameter input? (y/n): ").strip().lower()
            #if use_set_weights == 'n':
            #    weights = manual_adjust_weights(p_weights)
            #else:
            #    weights = p_weights
            return all_weight_configurations, weight_list_names, scenario_type, specific_weights

        else:
            return None, None, scenario_type, specific_weights, add_random, add_min
    #else:
        #add_random = None





def run_all_scenarios():
    save_folder_name = f"specific_weights_check_steps_S{S}_d{demand}_T{simulation_duration}"
    #save_folder_name = f"w{step_values}_S{S}_d{demand}_T{simulation_duration}_s{s_values}_cl{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}"
    os.makedirs(save_folder_name, exist_ok=True)
    global seeds

    demand_sku_distribution(S, demand, s_values, colors, class_i1l, class_i1r, save_folder_name)
    demand_sku_distribution(S, demand, s_values, colors, class_i2l, class_i2r, save_folder_name)

    demand_lists = manage_total_demand(save_folder_name)

    weights_lists, weight_names, scenario_type, specific_weights, add_random, add_min = manage_slotting_objective()

    start_demand_idx = 0        #0=Da, 1=Db, 2=Dc, 3=Dd, 4=De, 5=Df
    start_weight_idx = 0
    start_seed_idx = 0          #0=seed1, 1=seed2
    end_demand_idx = 5          # Ending demand index
    end_weight_idx = None  # Ending weight index (set this according to your requirements)
    end_seed_idx = None

    use_set_seeds = input(f"\nUse {seeds} seeds from parameter input? (y/n): ").strip().lower()
    if use_set_seeds == 'n':
        seeds = int(input(f"\nSet number of seeds to use (int): ").strip().lower())
    print(f"\nUse {seeds} seed(s)")

    #run optimisations
    for demand_idx, (demand_name, demand_list) in enumerate(zip(demand_list_names, demand_lists)):
        if demand_idx < start_demand_idx:
            continue
        if demand_idx > end_demand_idx:
            break
        if specific_weights == 'y':
            weights_lists, weight_names = generate_specific_weight_configurations(demand_name, add_min, add_random)
            for weight_idx, (weight_name, weights_list) in enumerate(zip(weight_names, weights_lists)):
                if demand_idx == start_demand_idx and weight_idx < start_weight_idx:
                    continue
                if end_weight_idx is not None and demand_idx == end_demand_idx and weight_idx > end_weight_idx:
                    break
                for seed in range(seeds):
                    if demand_idx == start_demand_idx and weight_idx == start_weight_idx and seed < start_seed_idx:
                        continue
                    if end_seed_idx is not None and demand_idx == end_demand_idx and weight_idx == end_weight_idx and seed >= end_seed_idx:
                        break
                    random.shuffle(demand_list)
                    #parameter_identifier = f'{demand_name}_{weight_name}_{seed + 1}__w12{weights_list[0], weights_list[1], weights_list[2], weights_list[3]}wd{weights_list[4][0], weights_list[4][1], weights_list[4][2]}_SKUs{S}_demand{demand}_Time{simulation_duration}_svalues{s_values}_classes{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}'
                    parameter_identifier = f'{demand_name}_{weight_name}_{seed + 1}'
                    print(f'\nRunning {scenario_type} slotting optimisation with demand {demand_name}, weights {weight_name} and seed {seed + 1} of {seeds}')
                    #save_folder_name = f"{demand_name}w{step_values}_S{S}_d{demand}_T{simulation_duration}"
                    #os.makedirs(save_folder_name, exist_ok=True)
                    run_optimization(scenario_type, [I, J, demand_list, S, P, B, V, penalty, weights_list], simulation_duration, parameter_identifier, save_folder_name)  # args are: (scenario_type, scenario_size, simulation_duration) => ('base'/'prio, 'test'/'large', 10/sec/None)
                    print(f'Finished {scenario_type} slotting optimisation with demand {demand_name}, weights {weight_name} and seed {seed + 1} of {seeds}')

                 #if add_random == 'y':
                 #   parameter_identifier = f'{name}{seed + 1}_random_SKUs{S}_demand{demand}_SimTime{simulation_duration}_svalues{s_values}_classes{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}'
                 #    print(f'\nRunning random slotting optimisation with {name}, seed {seed + 1} of {seeds}')
                 #   run_optimization('weighted', [I, J, demand_list, S, P, B, V, penalty, random_weights], simulation_duration, parameter_identifier)  # args are: (scenario_type, scenario_size, simulation_duration) => ('base'/'prio, 'test'/'large', 10/sec/None)
                 #   print(f'\nFinished random slotting optimisation with {name}, seed {seed + 1} of {seeds}')

run_all_scenarios()

def run_Gall_scenarios():
    save_folder_name = f"Gall_step12_S{S}_d{demand}_T{simulation_duration}"
    #save_folder_name = f"w{step_values}_S{S}_d{demand}_T{simulation_duration}_s{s_values}_cl{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}"
    os.makedirs(save_folder_name, exist_ok=True)
    global seeds

    demand_lists = manage_gall_demand(save_folder_name)
    demand_list_names = ['Dr', 'Dp']

    #create variable demand_list
    scenario_type = 'weighted'
    specific_weights = 'y'
    add_min = 'n'
    add_random = 'n'

    start_demand_idx = 0        #0=Dr, 1=Dp
    start_weight_idx = 0
    start_seed_idx = 0          #0=seed1, 1=seed2
    end_demand_idx = 5          # Ending demand index
    end_weight_idx = None  # Ending weight index (set this according to your requirements)
    end_seed_idx = None

    use_set_seeds = input(f"\nUse {seeds} seeds from parameter input? (y/n): ").strip().lower()
    if use_set_seeds == 'n':
        seeds = int(input(f"\nSet number of seeds to use (int): ").strip().lower())
    print(f"\nUse {seeds} seed(s)")

    #run optimisations
    for demand_idx, (demand_name, demand_list) in enumerate(zip(demand_list_names, demand_lists)):
        if demand_idx < start_demand_idx:
            continue
        if demand_idx > end_demand_idx:
            break
        if specific_weights == 'y':
            weights_lists, weight_names = generate_specific_weight_configurations(demand_name, add_min, add_random)
            for weight_idx, (weight_name, weights_list) in enumerate(zip(weight_names, weights_lists)):
                if demand_idx == start_demand_idx and weight_idx < start_weight_idx:
                    continue
                if end_weight_idx is not None and demand_idx == end_demand_idx and weight_idx > end_weight_idx:
                    break
                for seed in range(seeds):
                    if demand_idx == start_demand_idx and weight_idx == start_weight_idx and seed < start_seed_idx:
                        continue
                    if end_seed_idx is not None and demand_idx == end_demand_idx and weight_idx == end_weight_idx and seed >= end_seed_idx:
                        break
                    random.shuffle(demand_list)
                    #parameter_identifier = f'{demand_name}_{weight_name}_{seed + 1}__w12{weights_list[0], weights_list[1], weights_list[2], weights_list[3]}wd{weights_list[4][0], weights_list[4][1], weights_list[4][2]}_SKUs{S}_demand{demand}_Time{simulation_duration}_svalues{s_values}_classes{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}'
                    parameter_identifier = f'{demand_name}_{weight_name}_{seed + 1}'
                    print(f'\nRunning {scenario_type} slotting optimisation with demand {demand_name}, weights {weight_name} and seed {seed + 1} of {seeds}')
                    #save_folder_name = f"{demand_name}w{step_values}_S{S}_d{demand}_T{simulation_duration}"
                    #os.makedirs(save_folder_name, exist_ok=True)
                    run_optimization(scenario_type, [I, J, demand_list, S, P, B, V, penalty, weights_list], simulation_duration, parameter_identifier, save_folder_name)  # args are: (scenario_type, scenario_size, simulation_duration) => ('base'/'prio, 'test'/'large', 10/sec/None)
                    print(f'Finished {scenario_type} slotting optimisation with demand {demand_name}, weights {weight_name} and seed {seed + 1} of {seeds}')

                 #if add_random == 'y':
                 #   parameter_identifier = f'{name}{seed + 1}_random_SKUs{S}_demand{demand}_SimTime{simulation_duration}_svalues{s_values}_classes{class_i1l}-{class_i1r}-{class_i2l}-{class_i2r}_J{P}_B{B}_V{V}'
                 #    print(f'\nRunning random slotting optimisation with {name}, seed {seed + 1} of {seeds}')
                 #   run_optimization('weighted', [I, J, demand_list, S, P, B, V, penalty, random_weights], simulation_duration, parameter_identifier)  # args are: (scenario_type, scenario_size, simulation_duration) => ('base'/'prio, 'test'/'large', 10/sec/None)
                 #   print(f'\nFinished random slotting optimisation with {name}, seed {seed + 1} of {seeds}')

#run_Gall_scenarios()








