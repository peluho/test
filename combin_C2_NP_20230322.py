import datetime
import json
import os
import pathlib
import sys
import itertools
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

def read_input_csv(csv_path):
    """Read input CSV file and return data as a numpy array."""
    header_rows = 9
    dtype = {
        'names': [
            'ElemNo',
            'cas',
            'SX',
            'SY',
            'SXY',
            'MX',
            'MY',
            'MXY',
            'TZX',
            'TYZ',
            'SX_top',
            'SX_bot',
            'SY_top',
            'SY_bot',
            'Ep',
        ],
        'formats': [
            'int32',
            'int32',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
        ]
    }
    return np.genfromtxt(csv_path, delimiter=',', skip_header=header_rows, dtype=dtype)

# def read_lc_data(path_input, lc_names):
#     """Read LC data from CSV files and return as a dictionary."""
#     lc_data = {}
#     for lc_name in lc_names:
#         lc_path = pathlib.Path(path_input) / f'shell_{lc_name}.csv'
#         lc_data[lc_name] = {
#             'data': read_input_csv(lc_path),
#             'filePath': str(lc_path),
#             'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(lc_path)).strftime("%m/%d/%Y %H:%M:%S"),
#         }
#     return lc_data

def read_lc_data(path_input, lc_names):
    input_data = pd.read_excel(path_input, sheet_name=0, header=None)
    lc_data = []
    for lc in lc_names:
        # lc_info = input_data[input_data[0] == lc].values.tolist()[0][2:4]
        filtered_data = input_data[input_data[0] == lc].values.tolist()
        if filtered_data:
            lc_info = filtered_data[0][2:4]
        else:
            lc_info = ['', '']

        lc_data.append(lc_info)
    return lc_data

def write_output_csv(output_csv_path, lc_factors, lc_data):
    """Write LC combination to CSV file."""
    dtype = np.dtype({
        'names': [
            'ElemNo',
            'cas',
            'SX',
            'SY',
            'SXY',
            'MX',
            'MY',
            'MXY',
            'TZX',
            'TYZ',
            'SX_top',
            'SX_bot',
            'SY_top',
            'SY_bot',
            'Ep',
        ],
        'formats': [
            'int32',
            'int32',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
        ],
    })
    with open(output_csv_path, 'w') as f:
        combo_strs = ['', '', '']
        for lc_name, lc_factor in lc_factors.items():
            if lc_name.startswith('LC (Cryostat)'):
                combo_strs[0] += f"1 x [{lc_name}]"
                combo_strs[0] += f"1 x [{lc_names[lc_name]}]"
            else:
                combo_strs[0] += f"{lc_factor} x [{lc_name}]"
                combo_strs[1] += f"{lc_factor} x [{lc_nums[lc_name]}]"
                combo_strs[2] += f"{lc_factor:.3f} x [{lc_name}]"
                # Add LC data to the COMB dictionary
                lc_data = lc_datas[lc_name]
                for key, value in lc_data.items():
                    if key in COMB:
                        COMB[key] += value * lc_factor

            # Create string to show the combinations
            combo_str = f"{combo_strs[0]}\n{combo_strs[1]}\n{combo_strs[2]}"

            # Write to the output CSV file
            f.write(combo_str + '\n')
            f.write('!\n')
            for lc_name, lc_factor in lc_factors.items():
                lc_data = lc_datas[lc_name]
                f.write(f"! [{lc_name}], {lc_mod_dates[lc_name]}, {lc_files[lc_name]}\n")
            f.write('!\n')
            for header in [column_names, column_units]:
                header_str = '! '
                for col_value in header:
                    header_str += f"{col_value},"
                f.write(f"{header_str}\n")

            # Output data using previous definitions
            np.savetxt(f, COMB, fmt=fmt)  # delimiter=','

#
# import itertools
# import pandas as pd
#
# def generate_combination_csv(input_excel_file, sheet_number, output_dir):
#     # Read input data
#     input_data = pd.read_excel(input_excel_file, sheet_name=sheet_number, header=None)
#     lc_names = input_data.iloc[1:, 0].values.tolist()
#     lc_nums = input_data.iloc[1:, 1].values.tolist()
#     lc_datas = input_data.iloc[1:, 2].values.tolist()
#     lc_mod_dates = input_data.iloc[1:, 3].values.tolist()
#
#     # Remove empty values from lc_nums and lc_datas
#     lc_nums = [str(num) if pd.notna(num) else '' for num in lc_nums]
#     lc_datas = [data if pd.notna(data) else '' for data in lc_datas]
#
#     # Create LC factors dictionary
#     lc_factors = {}
#     for lc_name, lc_num, lc_data in zip(lc_names, lc_nums, lc_datas):
#         if pd.notna(lc_name) and lc_name not in lc_factors:
#             lc_factors[lc_name] = {'LC Number': lc_num, 'Data': lc_data}
#
#     # Generate all combinations
#     combos = list(itertools.product(*lc_factors.values()))
#
#     # Write to output CSV file
#     output_csv_path = f"{output_dir}/output.csv"
#     with open(output_csv_path, 'w') as f:
#         combo_strs = ['', '', '']
#         for lc_name, lc_factor in lc_factors.items():
#             if str(lc_name).startswith('LC (Cryostat)'):
#                 combo_strs[0] += f"1 x [{lc_name}]"
#                 combo_strs[1] += f"{lc_factor['Data']}"
#                 combo_strs[2] += f"{lc_factor['LC Number']}"
#
#         f.write(f"Description,Data,LC Number\n")
#         for combo in combos:
#             combo_strs[0] += f",1 x [{combo[0]['LC Name']}]"
#             combo_strs[1] += f",{combo[0]['Data']}"
#             combo_strs[2] += f",{combo[0]['LC Number']}"
#             f.write(f"{combo_strs[0]},{combo_strs[1]},{combo_strs[2]}\n")

def generate_combination_csv(input_excel_file, sheet_number, output_dir):
    # Read input data
    input_data = pd.read_excel(input_excel_file, sheet_name=sheet_number, header=None)
    lc_names = input_data.iloc[1:, 0].values.tolist()
    lc_nums = input_data.iloc[1:, 1].values.tolist()

    # Remove empty values from lc_nums
    lc_nums = [str(num) if pd.notna(num) else '' for num in lc_nums]

    # Create LC factors dictionary
    lc_factors = {}
    for lc_name, lc_num in zip(lc_names, lc_nums):
        if pd.notna(lc_name) and lc_name not in lc_factors:
            lc_data = read_lc_data(input_excel_file, [lc_name])[0]
            lc_factors[lc_name] = {'LC Number': lc_num, 'Data': lc_data}

    # Generate all combinations
    combos = list(itertools.product(*lc_factors.values()))

    # Write to output CSV file
    output_csv_path = f"{output_dir}/output.csv"
    with open(output_csv_path, 'w') as f:
        combo_strs = ['', '', '']
        for lc_name, lc_factor in lc_factors.items():
            print(lc_name)
            print(type(lc_name))
            if str(lc_name).startswith('LC (Cryostat)'):
                combo_strs[0] += f"1 x [{lc_name}]"
                combo_strs[1] += f"{lc_factor['Data'][0]}"
                combo_strs[2] += f"{lc_factor['LC Number']}"

        f.write(f"Description,Data,LC Number\n")
        for combo in combos:
            combo_strs[0] = ''
            combo_strs[1] = ''
            combo_strs[2] = ''
            for lc in combo:
                combo_strs[0] += f",1 x [{lc['LC Name']}]"
                combo_strs[1] += f",{lc['Data'][0]}"
                combo_strs[2] += f",{lc['LC Number']}"
            f.write(f"{combo_strs[0]},{combo_strs[1]},{combo_strs[2]}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_excel_file> <sheet_number>")
        return

    input_excel_file = sys.argv[1]
    sheet_number = int(sys.argv[2])
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_combination_csv(input_excel_file, sheet_number, output_dir)

if __name__ == '__main__':
    main()
