from openpyxl import load_workbook
import pandas as pd
import tracemalloc
import socket
import os
import time, datetime
from random import randint
from pathlib import Path
import numpy as np
import sys
sheetnumber=int(sys.argv[1])


tracemalloc.start()
time_start = datetime.datetime.now()
user = os.getlogin()

worksheet_names=list()
path_Excel = '\\\\io-ws-ccstore1.iter.org\\ANSYS_Data\\perezd\\100 CONSTRUCTION DESIGN V2\\903_ANSYS_2019\\Design_book_A2022\\' #Excel file with all the combinations for configuration 1
path_input = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\02_Config2\\02_LC\\Output\\Clusters_CSV\\LC_calcs\\' # path input load cases
path_output = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\02_Config2\\03_Combinations\\07_CSV\\01_WS\\' #The output will be generated in this folder

xl = pd.ExcelFile(os.path.join(path_Excel,'Appendix_2.xlsx'))
for sheet_name in xl.sheet_names:
    worksheet_names.append(sheet_name)

# workbook= pd.read_excel('Appendix_2.xlsx',sheet_name=[1,2,3,4,5,6,7,8],na_filter= False)
workbook= pd.read_excel(os.path.join(path_Excel,'Appendix_2.xlsx'),sheet_name=worksheet_names[sheetnumber],na_filter= False)


LC = {}
NEWC = {}
AUX = {}

LCName = {
    1: 'G+0.5Shr',
    # 1: 'G_permanent_load',
    4: 'Tqp_W',
    6: 'T_W',
    7: 'Q0_normal_variable_load',
    1000: 'EX1',
    2000: 'EY1',
    3000: 'EZ1',
    4000: 'EX2',
    5000: 'EY2',
    6000: 'EZ2',
    7000: 'EX4',
    8000: 'EY4',
    9000: 'EZ4',
    402: 'HIG_TK',
    30: 'GALL2_p',
    302: 'GALL2_p_b',
    322: 'DT_p_b',
}

LCName_SL3 = {i: f'LC{i}' for i in range(1500, 1578)}

#############
# Name and label columns, for input and output
columns = {
    "ElemNo": {'type': 'I', 'unit': ''},  
    "cas": {'type': 'I', 'unit': ''},  
    "SX": {'type': 'd', 'unit': 'N/m'},
    "SY": {'type': 'd', 'unit': 'N/m'},
    "SXY": {'type': 'd', 'unit': 'N/m'},
    "MX": {'type': 'd', 'unit': 'N.m/m'},
    "MY": {'type': 'd', 'unit': 'N.m/m'},
    "MXY": {'type': 'd', 'unit': 'N.m/m'},
    "TZX": {'type': 'd', 'unit': 'N/m'},
    "TYZ": {'type': 'd', 'unit': 'N/m'},
    "SX_top": {'type': 'd', 'unit': 'N/m²'},
    "SX_bot": {'type': 'd', 'unit': 'N/m²'},
    "SY_top": {'type': 'd', 'unit': 'N/m²'},
    "SY_bot": {'type': 'd', 'unit': 'N/m²'},
    "Ep": {'type': 'd', 'unit': 'm'},
    }
columnNames = []
columnTypes = []
columnUnits = []
fmt = ''
for columnName, column in columns.items():
    columnNames.append(columnName)
    columnTypes.append(column['type'])
    columnUnits.append(column['unit'])
    if column['type'] == 'I':
        fmt+= '%10.0f, '
    elif column['type'] == 'd':
        fmt+= '%15.7e, '


dtypecsv = np.dtype({'names': columnNames, 'formats': columnTypes})
for key, name in LCName.items():
    print('Reading load case {}: {}'.format(key, name))
    filePath = os.path.join(
                    path_input,
                    'shell_' + name + '.csv')
    if name=='G_permanent_load':
        LC[name] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                   'filePath':filePath,
                   'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime("%m/%d/%Y %H:%M:%S")}
    else:
        LC[name] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                   'filePath':filePath,
                   'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime("%m/%d/%Y %H:%M:%S")}


# for sheet_name in worksheet_names[1:8]:
# for sheet_name in worksheet_names[sheetnumber]:
#     sheet = workbook[sheetnumber]
sheet = workbook
columns = sheet.columns.values
size = sheet.shape
# if sheet_name=='SL+VDE_Beyond_basis_Cat_V':
# if sheet_name==worksheet_names[4]:
if sheetnumber==4:
    for key, name in LCName_SL3.items():
        print('Reading load case {}: {}'.format(key, name))
        filePath = os.path.join(
            path_input,
            'shell_LC' + str(key) + '.csv')
        LC[key] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                    'filePath': filePath,
                    'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime(
                        "%m/%d/%Y %H:%M:%S")}

for row in range(403,size[0]):
# for row in range(2,size[0]):

    # Dimension a large array for output - More efficient to preallocate in numpy
    COMB = np.empty(shape=(LC['G+0.5Shr']['data'].size,), dtype=dtypecsv)
    # print(LC['G+0.5Shr']['data'].size)
    # path = '\\\\io-ws-ccstore1.iter.org\\ANSYS_Data\\perezd\\100 CONSTRUCTION DESIGN V2\\903_ANSYS_2019\\Design_book_A2022\\'
    # if os.path.exists(os.path.join(path, str(worksheet_names[sheetnumber]))) == False:
    #     os.makedirs(os.path.join(path, str(worksheet_names[sheetnumber])))

    LCsUsed = []
    comboStrs = ['! ','! ','! ']
    for col in columns:
        if sheet[col][row] == '':
            continue
        if col == 'Factor':
            break
        if col != 'Comb':
            LCsUsed.append(col)
            if col == 'LC (Cryostat)' and sheet['LC (Cryostat)'][row] not in [LCName, LCName_SL3]:
                print('Reading load case {}: {}'.format(sheet['LC (Cryostat)'][row], sheet['LC (Cryostat)'][row]))
                filePath = os.path.join(
                    path_input,
                    'shell_LC' + str(sheet['LC (Cryostat)'][row]) + '.csv')
                LC['LC (Cryostat)'] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                          'filePath': filePath,
                          'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime(
                              "%m/%d/%Y %H:%M:%S")}

            #Create strings to show the combinations - Useful for debugging
            if comboStrs[0] != '! ':
                for j in [0,1,2]:
                    comboStrs[j] += '  +  '
            # worksheet[sheet][col][row]
            if col == 'LC (Cryostat)':
                lcfact = 1
                comboStrs[0] += '{factors} x [{comboName}]'.format(factors=1,
                                                                   comboName=col)
                comboStrs[1] += '{factors} x [{comboNum}]'.format(factors=1,
                                                                  comboNum=sheet[col][row])
            else:
                lcfact = eval(str(sheet[col][row]))
                comboStrs[0] += '{factors} x [{comboName}]'.format(factors=str(sheet[col][row]),
                                                                   comboName=col)  # LCName[col[i]]
                comboStrs[1] += '{factors} x [{comboNum}]'.format(factors=str(sheet[col][row]),
                                                                  comboNum=sheet[col][1])
            comboStrs[2] += '{evalFactor:.3f} x [{comboNum}]'.format(comboNum=col, evalFactor=lcfact)

            for item in ['SX', 'SY', 'SXY', 'MX', 'MY', 'MXY', 'TZX', 'TYZ', 'SX_top','SX_bot','SY_top','SY_bot']:
                COMB[item] += LC[col]['data'][item]*lcfact
            for item in ['ElemNo','Ep']:
                COMB[item] = LC[col]['data'][item]
            for item in ['cas']:
                COMB[item] = 0 #np.full((len(LC[1]),), rows2[14].value)

    #Create strings to show the combinations - Useful for debugging
    for j in [0,1,2]:
        comboStrs[j] = comboStrs[j].replace('  +  -', '  -  ')
    comboStr = comboStrs[0] + '\n' + comboStrs[1] + '\n' + comboStrs[2]
    print(comboStr)

    #################################################################
    if not os.path.exists(os.path.join(
                    path_output,worksheet_names[sheetnumber])):
        os.makedirs(os.path.join(
                    path_output,worksheet_names[sheetnumber]))
    with open(os.path.join(
                    path_output,worksheet_names[sheetnumber],'combin' + "{:07d}".format(sheet['Comb'][row]) + '.csv'), "w") as f:

        print('Writing file: {}'.format(f.name))
        #Create headers
        f.write(comboStr + "\n")
        f.write("!" + "\n")
        for LCNum in LCsUsed:
            f.write('! [{LCNum}], {date}, {filePath}\n'.format(LCNum=LCNum, date=LC[LCNum]['modifiedDate'], fmt=fmt, filePath=LC[LCNum]['filePath']))

        f.write("!" + "\n")
        for header in [columnNames,columnUnits]:
            headerStr = '! '
            for colValue in header:
                headerStr += colValue + ','
            f.write(headerStr + "\n")

        #Output data using previous definitions
        np.savetxt(f, COMB, fmt=fmt) #delimiter=','

    #For debug, Remove to all combos to be created
    #break