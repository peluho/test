import datetime
import json
import os
import pathlib
import sys

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

def read_lc_data(path_input, lc_names):
    """Read LC data from CSV files and return as a dictionary."""
    lc_data = {}
    for lc_name in lc_names:
        lc_path = pathlib.Path(path_input) / f'shell_{lc_name}.csv'
        lc_data[lc_name] = {
            'data': read_input_csv(lc_path),
            'filePath': str(lc_path),
            'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(lc_path)).strftime("%m/%d/%Y %H:%M:%S"),
        }
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

