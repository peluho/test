import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
import time

from typing import List, Tuple

# Define constants
# INPUT_FILE = '\\\\io-ws-ccstore1.iter.org\\ANSYS_Data\\perezd\\100 CONSTRUCTION DESIGN V2\\903_ANSYS_2019\\Design_book_A2022\\Appendix_2.xlsx'
SINGLE_LOAD_CASES_DIR = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\02_Config2\\02_LC\\Output\\Clusters_CSV\\LC_calcs\\' # path input load cases
# output_dir = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\02_Config2\\03_Combinations\\07_CSV\\01_WS\\' #The output will be generated in this folder

# Define a function to load the input file
import numpy as np
import pandas as pd
import numexpr as ne

def load_input_file(input_excel_file, sheet_number, output_dir):
    input_data = pd.read_excel(input_excel_file, sheet_name=sheet_number, header=None)
    lc_coeffs = {}
    for lc_position in range(len(input_data.columns)):
        lc_name = input_data.iloc[0, lc_position]
        coeffs = input_data.iloc[2:, lc_position].tolist()
        if lc_name is not np.nan:
            if all(pd.isna(x) or isinstance(x, (int, float, str)) for x in coeffs):
                lc_coeffs[lc_position] = (lc_name, [])
                for c in coeffs[1:]:
                    if isinstance(c, str):
                        try:
                            lc_coeffs[lc_position][1].append(ne.evaluate(c))
                        except:
                            lc_coeffs[lc_position][1].append(c)
                    elif isinstance(c, (int, float)):
                        lc_coeffs[lc_position][1].append(c)
            else:
                continue
    lc_coeffs = {k: v for k, v in lc_coeffs.items() if v[1]} # Remove empty values
    lc_names = [lc[0] for lc in lc_coeffs.values()]
    lc_nums = input_data.iloc[2, :].fillna(0.0).astype(float).tolist()
    return lc_coeffs, lc_names, lc_nums


import numexpr as ne
import sys

def generate_combinations(lc_coeffs, lc_nums):
    comb_list = []
    for key, value in lc_coeffs.items():
        if value[0] == 'Comb':
            comb_list = value[1]
            break

    combinations = []
    for row in range(0, len(comb_list)):
        LCsUsed = []
        comboStrs = ['! ', '! ', '! ']
        combo_factors = []
        for pos, (lc_name, coeffs) in lc_coeffs.items():
            if lc_name == 'Comb':
                continue
            if not pd.isna(lc_name) and lc_name != 0 and coeffs[row] != 0 and not pd.isna(coeffs[row]):
                if lc_name == 'LC (Cryostat)':
                    LCsUsed.append(coeffs[row])
                else:
                    LCsUsed.append(lc_name)
                combo_factors.append(coeffs[row])
                if comboStrs[0] != '! ':
                    for j in [0, 1, 2]:
                        comboStrs[j] += '  +  '
                if lc_name == 'LC (Cryostat)':
                    lcfact = 1
                    comboStrs[0] += '{factors} x [{comboName}]'.format(factors=lcfact,
                                                                       comboName=lc_name)
                    comboStrs[1] += '{factors} x [{comboNum}]'.format(factors=lcfact,
                                                                      comboNum=lc_nums[pos])
                    comboStrs[2] += '{evalFactors:.3f} x [{comboName}]'.format(evalFactors=lcfact,
                                                                               comboName=coeffs[row])
                    break  # Exit the loop
                else:
                    comboStrs[0] += '{factors} x [{comboName}]'.format(factors=str(coeffs[row]),
                                                                       comboName=lc_name)
                    comboStrs[1] += '{factors} x [{comboName}]'.format(factors=str(coeffs[row]),
                                                                       comboName=lc_nums[pos])
                    comboStrs[2] += '{evalFactors:.3f} x [{comboName}]'.format(evalFactors=coeffs[row],
                                                                           comboName=lc_nums[pos])
            # Create strings to show the combinations - Useful for debugging
        for j in [0, 1, 2]:
            comboStrs[j] = comboStrs[j].replace('  +  -', '  -  ')
        comboStr = comboStrs[0] + '\n' + comboStrs[1] + '\n' + comboStrs[2]
        if combo_factors:
            combinations.append((comb_list[row], combo_factors, comboStrs, LCsUsed))

    return combinations

# Define a function to load the single load cases
# def load_single_load_cases(lc_names, lc_nums, combinations):
#     lc_dict = dict(zip(lc_names, lc_nums))  # Merge lc_names with lc_nums
#     single_load_cases = {}
#     single_load_cases_changed = False  # Initialize flag for checking if single load cases have changed
#     for combo in combinations:
#         LCsUsed = combo[3]  # Extract LCsUsed from the combinations tuple
#         for lc_name in LCsUsed:
#             lc_num = lc_dict.get(lc_name)
#             if lc_num is None and isinstance(lc_name, int):  # Check if lc_name is a number and not in lc_dict
#                 lc_num = lc_name  # Use lc_name as lc_num
#                 filename = 'shell_LC' + str(lc_name) + '.csv'  # Use lc_name in the filename
#             else:
#                 filename = f"shell_{lc_name}.csv"
#             if lc_num is not None and lc_num != 0:
#                 filepath = os.path.join(SINGLE_LOAD_CASES_DIR, filename)
#                 if os.path.isfile(filepath):
#                     print(f"Reading load case {lc_num}: {lc_name}")
#                     lc_matrix = np.loadtxt(filepath, delimiter=',', skiprows=9)
#                     # Check if the single load case has changed
#                     if lc_name in single_load_cases:
#                         if not np.array_equal(lc_matrix, single_load_cases[lc_name]):
#                             single_load_cases_changed = True
#                     else:
#                         single_load_cases_changed = True
#                     single_load_cases[lc_name] = lc_matrix
#     # Check if any single load cases have been added or changed, and print message if they have
#     if single_load_cases_changed:
#         print("Single load cases have been added or changed.")
#     return single_load_cases


import os
import numpy as np
from multiprocessing import Pool

SINGLE_LOAD_CASES_DIR = "SingleLoadCases"

def read_single_load_case(args):
    lc_name, lc_num, filename = args
    filepath = os.path.join(SINGLE_LOAD_CASES_DIR, filename)
    if os.path.isfile(filepath):
        print(f"Reading load case {lc_num}: {lc_name}")
        lc_matrix = np.loadtxt(filepath, delimiter=',', skiprows=9)
        return lc_name, lc_matrix
    else:
        return lc_name, None

def load_single_load_cases(lc_names, lc_nums, combinations):
    print(f"lc_name={lc_names}, lc_num={lc_nums}")
    lc_dict = dict(zip(lc_names, list(lc_nums)))  # Merge lc_names with lc_nums
    single_load_cases = {}
    single_load_cases_changed = False  # Initialize flag for checking if single load cases have changed
    for combo in combinations:
        LCsUsed = combo[3]  # Extract LCsUsed from the combinations tuple
        for lc_name in LCsUsed:
            lc_num = lc_dict.get(lc_name)
            if lc_num is None and isinstance(lc_name, int):  # Check if lc_name is a number and not in lc_dict
                lc_num = lc_name  # Use lc_name as lc_num
                filename = 'shell_LC' + str(lc_name) + '.csv'  # Use lc_name in the filename
            else:
                filename = f"shell_{lc_name}.csv"
            if lc_num is not None and lc_num != 0:
                filepath = os.path.join(SINGLE_LOAD_CASES_DIR, filename)
                if os.path.isfile(filepath):
                    print(f"Reading load case {lc_num}: {lc_name}")
                    lc_matrix = np.loadtxt(filepath, delimiter=',', skiprows=9)
                    # Check if the single load case has changed
                    if lc_name in single_load_cases:
                        if not np.array_equal(lc_matrix, single_load_cases[lc_name]):
                            single_load_cases_changed = True
                    else:
                        single_load_cases_changed = True
                    single_load_cases[lc_name] = lc_matrix
    # Check if any single load cases have been added or changed, and print message if they have
    if single_load_cases_changed:
        print("Single load cases have been added or changed.")
    return single_load_cases

def write_output_file(combinations, single_load_cases, output_dir):
    # Write output file
    with open(output_dir, 'w') as f:
        # Write load case combinations
        f.write("Load case combinations:\n")
        for combo in combinations:
            f.write(f"{combo[0]}: {combo[2]}\n")
        f.write("\n")

        # Write single load cases
        f.write("Single load cases:\n")
        for lc_name, lc_matrix in single_load_cases.items():
            f.write(f"{lc_name}\n")
            np.savetxt(f, lc_matrix, delimiter=',', fmt='%.3f')
            f.write("\n")


def combine_load_cases(single_load_cases, lc_coeffs):
    combos = generate_combinations(lc_coeffs)
    all_load_cases = {}

    # Add single load cases to the combined load cases
    for lc_name, lc_matrix in single_load_cases.items():
        all_load_cases[lc_name] = lc_matrix

    # Combine load cases using the combination factors
    for combo in combos:
        combo_name = f"comb_{combos.index(combo)}"
        combo_matrix = np.zeros_like(all_load_cases[list(all_load_cases.keys())[0]])
        for lc in combo:
            lc_name = lc['LC Name']
            factor = lc['Coefficients'][0]
            lc_matrix = all_load_cases[lc_name]
            combo_matrix += factor * lc_matrix
        all_load_cases[combo_name] = combo_matrix

    return all_load_cases



# Define a function to combine the single load cases using the load combinations
def combine_single_load_cases(single_load_cases, combo_strs):
    for i, combo_str in enumerate(combo_strs):
        combo_matrix = np.zeros_like(list(single_load_cases.values())[0])
        for lc_name in combo_str.split(' + '):
            lc_factor, lc_name = lc_name.split(' x ')
            lc_matrix = single_load_cases[lc_name]
            combo_matrix += float(lc_factor) * lc_matrix
        np.savetxt(os.path.join(OUTPUT_DIR, f"combo_{i}.csv"), combo_matrix, delimiter=',')

# Define the main function
def main():
    if len(sys.argv) != 4:
        print("Usage: python combin_C2_NP.py <input_excel_file> <sheet_number> <output_csv_file>")
        return

    input_excel_file = sys.argv[1]
    sheet_number = int(sys.argv[2])
    output_dir = sys.argv[3]

    # Load the input file
    print(f"Reading the {input_excel_file}...")
    lc_coeffs, lc_names, lc_nums = load_input_file(input_excel_file, sheet_number, output_dir)

    # Generate the load combinations
    print("Generating load case combinations...")
    combinations = generate_combinations(lc_coeffs, lc_nums)

    # Load the single load cases
    # Create a list of indices corresponding to the load cases to be used
    lc_indices = []
    for i, name in enumerate(lc_names):
        if name in LCsUsed:
            lc_indices.append(i)
        elif isinstance(LCsUsed, int) and int(lc_nums[i]) == LCsUsed:
            lc_indices.append(i)

    # Create lists of load case names and numbers corresponding to the selected indices
    lc_names_selected = [lc_names[i] for i in lc_indices]
    lc_nums_selected = [lc_nums[i] for i in lc_indices]

    # Load the single load cases using the selected names and numbers
    print("Loading the single load cases...")
    with multiprocessing.Pool() as p:
        single_load_cases = dict(
            p.starmap(load_single_load_cases,
                      zip(lc_names_selected, lc_nums_selected, [combinations] * len(lc_names_selected))))

    # Write the output file
    print("Writing the output file...")
    write_output_file(combinations, single_load_cases, output_dir, binary=True)

    print("Done.")


if __name__ == '__main__':
    main()
