import math
import numpy as np

# Define the input parameters
data = {
    'h': [1.3, 0.455],
    'bw': [1.5, 1.5],
    'fck': [90, 90],
    'fys': [500, 500],
    'fs': [1.15, 1.15],
    'fc': [1.5, 1.5],
    'ds': [16, 14],
    'ss': [100, 100],
    'Ast': [60.32, 15.39],
    'dtz': [14, 14],
    'stz': [200, 200],
    'Azt': [69.3, 38.5],
    'dty': [14, 14],
    'sty': [200, 200],
    'Ayt': [30.8, 7.7],
    'alpha_s': [33.69 * math.pi/180, 45 * math.pi/180],
    'b_span': [2.690, 1.40]
}

def calculate_beam_data(data):
    Acro = []
    peri = []
    eff_wt = []
    eff_wt_aux = []
    eff_wt_aux.append([0.001 * (35 + 16 + 40 + 16 / 2), 0.001 * (35 + 16 / 2)])
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

    for jj in range(2):
        # Acro
        Acro.append([data['h'][jj] * data['bw'][jj], 0])
        # peri
        peri.append([2 * (data['h'][jj] + data['bw'][jj]), 0])
        # eff_wt
        if Acro[jj][0] / peri[jj][0] > eff_wt_aux[jj][0]:
            eff_wt.append([Acro[jj][0] / peri[jj][0], 0])
        else:
            eff_wt.append([eff_wt_aux[jj][0], 0])
        # peritor
        peritor.append([2 * ((data['h'][jj] - 2 * eff_wt[jj][0]) + ((data['bw'][jj] - 2 * eff_wt[jj][0]))), 0])
        # effz
        effz.append([0.9 * data['h'][jj], 0])
        # levz
        levz.append([0.9 * effz[jj][0], 0])
        # effy
        effy.append([0.9 * data['bw'][jj], 0])
        # levy
        levy.append([0.9 * effy[jj][0], 0])
        # Aenc
        Aenc.append([(data['h'][jj] - eff_wt[jj][0]) * (data['bw'][jj] - eff_wt[jj][0]), 0])
        # cotant
        maxctany.append([data['b_span'][jj] / data['bw'][jj], 0])
        maxctanz.append([data['b_span'][jj] / data['h'][jj], 0])
        if (1 / math.tan(data['alpha_s'][jj])) < maxctany[jj][0] and (1 / math.tan(data['alpha_s'][jj])) < maxctanz[jj][0]:
            if (1 / math.tan(data['alpha_s'][jj])) > 1:
                cotant.append([1 / math.tan(data['alpha_s'][jj]), 0])
            else:
                cotant.append([1, 0])
        elif maxctany[jj][0] < maxctanz[jj][0]:
            if maxctany[jj][0] > 1:
                cotant.append([maxctany[jj][0], 0])
            else:
                cotant.append([1, 0])
        else:
            if maxctanz[jj][0] > 1:
                cotant.append([maxctanz[jj][0], 0])
            else:
                cotant.append([1, 0])
        # fywd

    # fywd
    v_oper.append(data['fys'][jj] / data['fs'][jj])
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

# call the calculate_beam_data function
results = calculate_beam_data(data)

#Print the results
print(results)

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
