import math
import numpy as np

def calculate_beam_data(data):
    Acro = []
    peri = []
    eff_wt = []
    eff_wt_aux = []
    peritor = []
    effz = []
    levz = []
    effy = []
    levy = []
    Aenc = []
    cotant = []
    maxctany = []
    maxctanz = []
    fywd = []
    fyd = []
    fcd = []
    fcm = []
    fctm = []
    v1 = []
    acw = []
    v = []
    v_oper = []

    for jj in range(2):
        # Acro
        Acro.append([data["h"][jj] * data["bw"][jj], 0])
        # peri
        peri.append([2 * (data["h"][jj] + data["bw"][jj]), 0])
        # eff_wt
        eff_wt_aux.append([0.001 * (35 + 16 + 40 + 16 / 2), 0.001 * (35 + 16 / 2)])
        if Acro[jj][0] / peri[jj][0] > eff_wt_aux[jj][0]:
            eff_wt.append([Acro[jj][0] / peri[jj][0], 0])
        else:
            eff_wt.append([eff_wt_aux[jj][0], 0])
        # peritor
        peritor.append([2 * ((data["h"][jj] - 2 * eff_wt[jj][0]) + ((data["bw"][jj] - 2 * eff_wt[jj][0]))), 0])
        # effz
        effz.append([0.9 * data["h"][jj], 0])
        # levz
        levz.append([0.9 * effz[jj][0], 0])
        # effy
        effy.append([0.9 * data["bw"][jj], 0])
        # levy
        levy.append([0.9 * effy[jj][0], 0])
        # Aenc
        Aenc.append([(data["h"][jj] - eff_wt[jj][0]) * (data["bw"][jj] - eff_wt[jj][0]), 0])
        # cotant
        maxctany.append([data["b_span"][jj] / data["bw"][jj], 0])
        maxctanz.append([data["b_span"][jj] / data["h"][jj], 0])
        if (1 / math.tan(data['alpha_s'][jj])) < data['b_span'][jj] / data['bw'][jj] and \
                (1 / math.tan(data['alpha_s'][jj])) < data['b_span'][jj] / data['h'][jj]:
            if 1 / math.tan(data['alpha_s'][jj]) > 1:
                cotant.append([1 / math.tan(data['alpha_s'][jj]), 0])
            else:
                cotant.append([1, 0])
        elif data['b_span'][jj] / data['bw'][jj] < data['b_span'][jj] / data['h'][jj]:
            if data['b_span'][jj] / data['bw'][jj] > 1:
                cotant.append([data['b_span'][jj] / data['bw'][jj], 0])
            else:
                cotant.append([1, 0])
        else:
            if data['b_span'][jj] / data['h'][jj] > 1:
                cotant.append([data['b_span'][jj] / data['h'][jj], 0])
            else:
                cotant.append([1, 0])
        # fywd
        v_oper.append([data['fys'][jj] / data['fs'][jj],0])
        fyd.append([data['fys'][jj] / data['fs'][jj], 0])
        fywd.append([min(fyd[jj][0], 0.8 * data['fys'][jj]), 0])
        # fcd
        fcd.append([data['fck'][jj] / data['fc'][jj], 0])
        # fcm
        fcm.append([data['fck'][jj] + 8, 0])
        # fctm
        if data['fck'][jj] <= 50:
            fctm.append([0.3 * data['fck'][jj], 0])
        else:
            fctm.append([2.12 * math.log(1 + fcm[jj][0] / 10), 0])
        # v1
        if 0.9 - data['fck'][jj] / 200 > 0.5:
            v1.append([0.9 - data['fck'][jj] / 200, 0])
        else:
            v1.append([0.5, 0])
        # alphacw
        acw.append([1, 0])
        # v
        v.append([0.6 * (1 - data['fck'][jj] / 250), 0])
    # Create dictionaries for deep and shallow beam data
    deep_beam_data = {
        'Acro': Acro[0],
        'peri': peri[0],
        'eff_wt': eff_wt[0],
        'eff_wt_aux': eff_wt_aux[0],
        'peritor': peritor[0],
        'effz': effz[0],
        'levz': levz[0],
        'effy': effy[0],
        'levy': levy[0],
        'Aenc': Aenc[0],
        'cotant': cotant[0],
        'maxctany': maxctany[0],
        'maxctanz': maxctanz[0],
        'fywd': fywd[0],
        'fyd': fyd[0],
        'fcd': fcd[0],
        'fcm': fcm[0],
        'fctm': fctm[0],
        'v1': v1[0],
        'acw': acw[0],
        'v': v[0],
        'v_oper': v_oper[0]
    }

    shallow_beam_data = {
        'Acro': Acro[1],
        'peri': peri[1],
        'eff_wt': eff_wt[1],
        'eff_wt_aux': eff_wt_aux[1],
        'peritor': peritor[1],
        'effz': effz[1],
        'levz': levz[1],
        'effy': effy[1],
        'levy': levy[1],
        'Aenc': Aenc[1],
        'cotant': cotant[1],
        'maxctany': maxctany[1],
        'maxctanz': maxctanz[1],
        'fywd': fywd[1],
        'fyd': fyd[1],
        'fcd': fcd[1],
        'fcm': fcm[1],
        'fctm': fctm[1],
        'v1': v1[1],
        'acw': acw[1],
        'v': v[1],
        'v_oper': v_oper[1]
    }

    # Return a dictionary containing deep and shallow beam data
    return {'deep_beam_data': deep_beam_data, 'shallow_beam_data': shallow_beam_data}

#Function to read the excel files

import pandas as pd

def read_csv_file(filename, comment_char, columns):
    """
    Reads a CSV file and returns a dataframe with selected columns.

    filename: str - the name of the CSV file to read.
    comment_char: str - the character used to comment out rows to skip.
    columns: list - a list of column names to select from the file.

    returns: pandas.DataFrame - a dataframe with the selected columns.
    """
    # read the entire CSV file into a dataframe
    df = pd.read_csv(filename, header=None, encoding='ISO-8859-1', comment= comment_char)

    # create a name mapping dictionary based on the desired column names
    name_mapping = {i: columns[i]['name'] for i in range(len(columns))}
    # name_mapping = {}
    # for i, col_dict in columns.items():
    #     if 'name' in col_dict:
    #         name_mapping[i] = col_dict['name']

    # rename the columns based on the mapping
    df = df.rename(columns=name_mapping)

    # select the specified columns
    selected_columns = [col for col in columns.keys()]
    df = df.iloc[:, selected_columns]

    # return the resulting dataframe
    return df

def add_columns(df, beam_data, deep_beam_elements, shallow_beam_elements):
    """
    Adds new columns to the dataframe.

    df: pandas.DataFrame - the dataframe to modify.

    returns: pandas.DataFrame - the modified dataframe.
    """

    # Filter dataframe based on beam element number
    if df.iloc[0]['ElemNo'] in deep_beam_elements:
        Acro = beam_data['Acro'][0][0]
    elif df.iloc[0]['ElemNo'] in shallow_beam_elements:
        Acro = beam_data['Acro'][1][0]
    else:
        raise ValueError('Invalid beam element')

    # calculate the new N column
    df['N'] = np.where(df['N_OR'] < 0, np.minimum(df['N_OR'], df['N_EX']), np.maximum(df['N_OR'], df['N_EX']))

    # calculate the Tyy column
    df['Tyy'] = np.maximum(np.abs(df['TY_OR']), np.abs(df['TY_EX']))

    # calculate the Tzz column
    df['Tzz'] = np.maximum(np.abs(df['TZ_OR']), np.abs(df['TZ_EX']))

    # calculate the Tors column
    df['Tors'] = np.maximum(np.abs(df['TORS_OR']), np.abs(df['TORS_EX']))

    # Mean compressive stress (MPa)
    for i, data in enumerate([beam_data['deep_beam_data'], beam_data['shallow_beam_data']]):
        df[f'Sigma_cp_{i}'] = 0.001 * df['N'] / data['Acro'][0]

    # return the modified dataframe
    return df

def filter_beam_type(df, deep_beam_elements, shallow_beam_elements):
    """
    Filters the dataframe based on element number for deep and shallow beams.

    df: pandas.DataFrame - the input dataframe to filter.
    deep_beam_elements: list - a list of element numbers for deep beams.
    shallow_beam_elements: list - a list of element numbers for shallow beams.

    returns: tuple - a tuple of two dataframes, one for deep beams and one for shallow beams.
    """
    # filter the dataframe by the element numbers for deep beams
    deep_beam_df = df[df['ElemNo'].isin(deep_beam_elements)]

    # filter the dataframe by the element numbers for shallow beams
    shallow_beam_df = df[df['ElemNo'].isin(shallow_beam_elements)]

    # return the two filtered dataframes as a tuple
    return deep_beam_df, shallow_beam_df


# def calculate_beam_properties(df, beam_type):
    # """
    # Calculates the properties of a beam based on the given input data and beam type (deep or shallow).
    #
    # Args:
    #     df (pandas.DataFrame): Input data for the beam.
    #     beam_type (str): Type of beam. Should be either "deep" or "shallow".
    #
    # Returns:
    #     pandas.DataFrame: Output data with the calculated properties of the beam.
    # """
    # # Create output dataframe with column names
    # columns = ['B', 'H', 'fyd', 'fcd', 'v', 'levy', 'levz', 'alpha_s', 'cotant', 'Acro', 'Ted', 'Ast_t', 'Ast_l', 'Trd_max', 'Trd_max_iter']
    # df_out = pd.DataFrame(columns=columns)
    #
    # # Copy over input data
    # df_out.loc[0] = df.iloc[0]
    #
    # if beam_type == "deep":
    #     for i, row in df.iterrows():
    #         # Copy over input data
    #         df_out.iloc[i, :len(row)] = row.values
    #
    #         # Calculate deep beam properties
    #         df_out.at[i, 'Ted'] = np.where(row['Nd'] < 0, np.minimum(row['Nd'], row['Nc']) * 10, np.maximum(row['Nd'], row['Nc']) * 10)
    #         df_out.at[i, 'Ast_t'] = 10 * row['T'] / (df_out.at[0, 'Acro'] * row['levz'] * np.tan(row['alpha_s']))
    #         df_out.at[i, 'Ast_l'] = 10 * row['T'] * np.tan(row['alpha_s']) / (df_out.at[0, 'Acro'] * row['levz'] * df_out.at[0, 'fyd'])
    #         df_out.at[i, 'Trd_max'] = 2 * df_out.at[0, 'v'] * df_out.at[0, 'acw'] * df_out.at[0, 'fcd'] * df_out.at[0, 'Acro'] * df_out.at[0, 'tef'] * np.sin(row['alpha_s']) * np.cos(row['alpha_s']) * 1000
    #         df_out.at[i, 'Trd_max_iter'] = 1000 * 0.068 * df_out.at[0, 'h'] * row['levy'] * (1 - row['cotant'] / 4 if row['Ted'] < 0 else 1 - 0.36)
    # elif beam_type == "shallow":
    #     for i, row in df.iterrows():
    #         # Copy over input data
    #         df_out.iloc[i, :len(row)] = row.values
    #
    #         # Calculate shallow beam properties
    #         df_out.at[i, 'Ted'] = np.where(row['Nd'] < 0, np.minimum(row['Nd'], row['Nc']) * 10, np.maximum(row['Nd'], row['Nc']) * 10)
    #         df_out.at[i, 'As'] = 10 * row['T'] / (df_out.at[0, 'Acro'] * row['levz'] * np.tan(row['alpha_s']))
    #         df_out.at[i, 'Trd_max'] = 2 * df_out.at[0, 'v'] * df_out.at[0, 'acw'] * df_out.at[0, 'fcd'] * df_out.at[0, 'Acro'] * df_out.at[0, 'tef'] * np.sin(row['alpha_s']) * np.cos(row['alpha_s']) * 1000
    #         df_out.at[i, 'Trd_max_iter'] = 1000 * 0.068 * df_out.at[0, 'h'] * row['levy'] * (1 - row['cotant'] / 4) if row['Ted'] < 0 else 1000 * 0.068 * df_out.at[0, 'h'] * row['levy'] * (1 - 0.36)
    # else:
    #     raise ValueError("Invalid beam type")
    #
    # return df_out

def main():

    data = {
        "h": [1.3, 0.455],
        "bw": [1.5, 1.5],
        "fck": [90, 90],
        "fys": [500, 500],
        "fs": [1.15, 1.15],
        "fc": [1.5, 1.5],
        "ds": [16, 14],
        "ss": [100, 100],
        "Ast": [60.32, 15.39],
        "dtz": [14, 14],
        "stz": [200, 200],
        "Azt": [69.3, 38.5],
        "dty": [14, 14],
        "sty": [200, 200],
        "Ayt": [30.8, 7.7],
        "alpha_s": [33.69 * math.pi / 180, 45 * math.pi / 180],
        "b_span": [2.69, 1.4],
    }

    # call the calculate_beam_data function
    results = calculate_beam_data(data)

    #Print the results
    print(results)

    # Load input data
    filename = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\02_Config2\\03_Combinations\\07_CSV\\01_WS\\VDE_Normal_Cat_II\\beam_combin0803101.csv'
    comment_char = '!'
    columns = {
        0: {'name': 'ElemNo', 'unit': ''},
        1: {'name': 'cas', 'unit': ''},
        2: {'name': 'N_OR', 'unit': 'N'},
        3: {'name': 'TY_OR', 'unit': 'N'},
        4: {'name': 'TZ_OR', 'unit': 'N'},
        5: {'name': 'N_EX', 'unit': 'N'},
        6: {'name': 'TY_EX', 'unit': 'N'},
        7: {'name': 'TZ_EX', 'unit': 'N'},
        8: {'name': 'TORS_OR', 'unit': 'N.m'},
        9: {'name': 'MZ_OR', 'unit': 'N.m'},
        10: {'name': 'MY_OR', 'unit': 'N.m'},
        11: {'name': 'TORS_EX', 'unit': 'N.m'},
        12: {'name': 'MZ_EX', 'unit': 'N.m'},
        13: {'name': 'MY_EX', 'unit': 'N.m'},
        14: {'name': 'A', 'unit': 'm2'},
        15: {'name': 'IZ', 'unit': 'm4'},
        16: {'name': 'IY', 'unit': 'm4'}
    }

    # Define beam element lists
    deep_beam_elements = [260742, 260743, 260744, 260745, 260746, 260747, 260750, 260751, 260752, 260753, 260754,
                          260755,
                          260758, 260759, 260760, 260761, 260762, 260763, 260766, 260767, 260768, 260769, 260770,
                          260771,
                          260774, 260775, 260776, 260777, 260778, 260779, 260782, 260783, 260784, 260785, 260786,
                          260787,
                          260790, 260791, 260792, 260793, 260794, 260795, 260798, 260799, 260800, 260801, 260802,
                          260803,
                          260806, 260807, 260808, 260809, 260810, 260811, 260814, 260815, 260816, 260817, 260818,
                          260819,
                          260822, 260823, 260824, 260825, 260826, 260827, 260830, 260831, 260832, 260833, 260834,
                          260835]

    shallow_beam_elements = [260721, 260722, 260723, 260724, 260725, 260726, 260727, 260728, 260729, 260730, 260731,
                             260732,
                             260733, 260734, 260735, 260736, 260737, 260738, 260739, 260740]

    # Read CSV file and filter for beam elements
    df = read_csv_file(filename, comment_char, columns)

    # Calculate additional columns
    beam_data = calculate_beam_data(data)
    # deep_beam_df = add_columns(df,beam_data)

    # Filter the DF based on element numbers
    # df = filter_beam_type(df, deep_beam_elements, shallow_beam_elements)
    deep_beam_df = add_columns(df, beam_data, deep_beam_elements, [])
    shallow_beam_df = add_columns(df, beam_data, [], shallow_beam_elements)

    # Print resulting dataframe
    print(df)

    # Calculate deep beam properties
    df_deep = df[df['beam_type'] == "deep"]
    df_deep_props = calculate_beam_properties(df_deep, "deep")

    # Calculate shallow beam properties
    df_shallow = df[df['beam_type'] == "shallow"]
    df_shallow_props = calculate_beam_properties(df_shallow, "shallow")

    # Combine results into a single dataframe
    df_props = pd.concat([df_deep_props, df_shallow_props])

    # Write results to output file
    df_props.to_excel("output_data.xlsx", index=False)



if __name__ == '__main__':
    main()


# def calculate_shear_reinforcement(data):
#     # Extract the required data from the input data
#     shear_force = data['Shear force (kN)'].values[0]
#     concrete_area = data['Concrete area (mm2)'].values[0]
#     concrete_strength = data['Concrete strength (MPa)'].values[0]
#     beam_width = data['Beam width (mm)'].values[0]
#     effective_depth = data['Effective depth (mm)'].values[0]
#     bar_diameter = data['Bar diameter (mm)'].values[0]
#     bar_spacing = data['Bar spacing (mm)'].values[0]
#
#     # Calculate the required shear reinforcement
#     alpha_v = 0.5 + 0.25 * (beam_width - bar_diameter) / bar_spacing
#     v_rdc = 0.27 * math.sqrt(concrete_strength) * (100 * bar_diameter / bar_spacing - 1) * bar_diameter / 1000
#     v_min = 0.035 * math.sqrt(concrete_strength) * effective_depth / 1000
#     v_ed = shear_force * 1000 / (alpha_v * concrete_area)
#
#     if v_ed <= v_rdc:
#         return 0
#     else:
#         return max(v_min, v_ed) * concrete_area / (0.9 * bar_diameter)
#
#
# def check_input_data(data):
#     # Check the input data for errors
#     errors = []
#
#     # Check the shear force
#     if data['Shear force (kN)'].values[0] <= 0:
#         errors.append('Shear force must be greater than 0.')
#
#     # Check the concrete area
#     if data['Concrete area (mm2)'].values[0] <= 0:
#         errors.append('Concrete area must be greater than 0.')
#
#     # Check the concrete strength
#     if data['Concrete strength (MPa)'].values[0] <= 0:
#         errors.append('Concrete strength must be greater than 0.')
#
#     # Check the beam width
#     if data['Beam width (mm)'].values[0] <= 0:
#         errors.append('Beam width must be greater than 0.')
#
#     # Check the effective depth
#     if data['Effective depth (mm)'].values[0] <= 0:
#         errors.append('Effective depth must be greater than 0.')
#
#     # Check the bar diameter
#     if data['Bar diameter (mm)'].values[0] <= 0:
#         errors.append('Bar diameter must be greater than 0.')
#
#     # Check the bar spacing
#     if data['Bar spacing (mm)'].values[0] <= 0:
#         errors.append('Bar spacing must be greater than 0.')
#
#     if errors:
#         # If there are errors, raise an exception with the error messages
#         raise ValueError('\n'.join(errors))
#
#
# # Read the input file
# data = pd.read_excel('path/to/input_file.xlsx')
#
# try:
#     check_input_data(data)
#     shear_reinforcement = calculate_shear_reinforcement(data)
#     print(shear_reinforcement)
# except ValueError as e:
#     # If there are errors, print the error message
#     print('Input data is invalid:')
#     print(str(e))
