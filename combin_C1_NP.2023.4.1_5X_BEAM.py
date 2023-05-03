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
path_input = '\\\\io-ws-ccstore1\\03.Ansys\\01_5X\\01_Config1\\02_LC\\Output\\LC_calcs' # path input load cases
path_output = '\\\\io-ws-ccstore1\\03.Ansys\\01_5X\\01_Config1\\03_Combinations\\07_CSV\\01_WS' #The output will be generated in this folder



xl = pd.ExcelFile(os.path.join(path_Excel,'Design_Bases_V05.0_Complete.xlsx'))
for sheet_name in xl.sheet_names:
    worksheet_names.append(sheet_name)

workbook= pd.read_excel(os.path.join(path_Excel,'Design_Bases_V05.0_Complete.xlsx'),skiprows=3,sheet_name=worksheet_names[sheetnumber],na_filter= False)
columns = workbook.columns.values


LC = {}
NEWC = {}
AUX = {}

if sheetnumber==1:
    LCName = {
        1: 'G+0.5Shr',
        2: 'Ge',
        4: 'Tqp_W',
        5: 'T_S',
        6: 'T_W',
        7: 'Q0_normal_variable_load',
        8: 'Qe',
        101: 'Crane_load_101',
        102: 'Crane_load_102',
        103: 'Crane_load_103',
        104: 'Crane_load_104',
        105: 'Crane_load_105',
        106: 'Crane_load_106',
        107: 'Crane_load_107',
        108: 'Crane_load_108',
        109: 'Crane_load_109',
        110: 'Crane_load_110',
        111: 'Crane_load_111',
        112: 'Crane_load_112',
        113: 'Crane_load_113',
        114: 'Crane_load_114',
        115: 'Crane_load_115',
        116: 'Crane_load_116',
        117: 'Crane_load_117',
        118: 'Crane_load_118',
        119: 'Crane_load_119',
        120: 'Crane_load_120',
        121: 'Crane_load_121',
        122: 'Crane_load_122',
        123: 'Crane_load_123',
        124: 'Crane_load_124',
        125: 'Crane_load_125',
        126: 'Crane_load_126',
        127: 'Crane_load_127',
        128: 'Crane_load_128',
        129: 'Crane_load_129',
        130: 'Crane_load_130',
        131: 'Crane_load_131',
        132: 'Crane_load_132',
        133: 'Crane_load_133',
        134: 'Crane_load_134',
        135: 'Crane_load_135',
        136: 'Crane_load_136',
        137: 'Crane_load_137',
        138: 'Crane_load_138',
        21: 'PC1LOC_p',
        22: 'PC2LOC_p',
        23: 'PC3LOC_p',
        24: 'PC4LOC_p',
        25: 'PC1LOV_p',
        26: 'PC2LOV_p',
        27: 'PC3LOV_p',
        28: 'PC4LOV_p',
        29: 'VAULT_p',
        30: 'GALL2_p',
        31: 'GALL1_p',
        32: 'DT_p',
        33: 'NBCELL_p',
        61: 'ISS',
        62: 'WDS',
        63: 'Corridor_Tritium',
        64: 'General_Tritium',
        65: 'HDT1',
        66: 'HDT2',
        67: 'HDT3',
        68: 'HDT4',
        69: 'HDT5',
        70: 'HDT6',
        71: 'HDT7',
        72: 'HDT8',
        73: 'HDT9',
        91: 'NBCELLdust_p',
    }
elif sheetnumber ==2:
    LCName = {
        1: 'G+0.5Shr',
        4: 'Tqp_W',
        7: 'Q0_normal_variable_load',
        21: 'PC1LOC_p',
        22: 'PC2LOC_p',
        23: 'PC3LOC_p',
        24: 'PC4LOC_p',
        25: 'PC1LOV_p',
        26: 'PC2LOV_p',
        27: 'PC3LOV_p',
        28: 'PC4LOV_p',
        29: 'VAULT_p',
        30: 'GALL2_p',
        31: 'GALL1_p',
        32: 'DT_p',
        33: 'NBCELL_p',
        34: 'PC1_LOCt',
        35: 'PC2_LOCt',
        36: 'CSR_t',
        37: 'NBCELL_t',
        38: 'Vault_T300s',
        39: 'Vault_T1200s',
        40: 'Vault_T3600s',
        41: 'AeexpN_External_Explosion_North',
        42: 'AeexpS_External_Explosion_South',
        43: 'AeexpE_External_Explosion_East',
        44: 'AeexpW_External_Explosion_West',
        461: 'Accidental_internal_flooding1',
        462: 'Accidental_internal_flooding2',
        463: 'Accidental_internal_flooding3',
        47: 'AT_S',
        48: 'AT_W',
        51: {8001 : 'Aapc_AirplaneCrash_1a',8002 : 'Aapc_AirplaneCrash_1b',8003 : 'Aapc_AirplaneCrash_1c',8004 : 'Aapc_AirplaneCrash_1d',8005 : 'Aapc_AirplaneCrash_1e',8006 : 'Aapc_AirplaneCrash_1f',8007 : 'Aapc_AirplaneCrash_1g',8008 : 'Aapc_AirplaneCrash_1h',8009 : 'Aapc_AirplaneCrash_1i'},
        52: {8010: 'Aapc_AirplaneCrash_2a', 8011: 'Aapc_AirplaneCrash_2b', 8012: 'Aapc_AirplaneCrash_2c', 8013: 'Aapc_AirplaneCrash_2d', 8014: 'Aapc_AirplaneCrash_2e', 8015: 'Aapc_AirplaneCrash_2f', 8016: 'Aapc_AirplaneCrash_2g', 8017: 'Aapc_AirplaneCrash_2h', 8018 : 'Aapc_AirplaneCrash_2i'},
        53: {8019: 'Aapc_AirplaneCrash_3a', 8020: 'Aapc_AirplaneCrash_3b', 8021: 'Aapc_AirplaneCrash_3c', 8022: 'Aapc_AirplaneCrash_3d', 8023: 'Aapc_AirplaneCrash_3e', 8024: 'Aapc_AirplaneCrash_3f', 8025: 'Aapc_AirplaneCrash_3g', 8026: 'Aapc_AirplaneCrash_3h', 8027 : 'Aapc_AirplaneCrash_3i'},
        54: {8028: 'Aapc_AirplaneCrash_4a1', 8029: 'Aapc_AirplaneCrash_4a2', 8030: 'Aapc_AirplaneCrash_4b', 8031: 'Aapc_AirplaneCrash_4c', 8032: 'Aapc_AirplaneCrash_4d1', 8033: 'Aapc_AirplaneCrash_4d2', 8034: 'Aapc_AirplaneCrash_4e', 8035: 'Aapc_AirplaneCrash_4f', 8036: 'Aapc_AirplaneCrash_4g', 8037: 'Aapc_AirplaneCrash_4h', 8038: 'Aapc_AirplaneCrash_4i', 8039: 'Aapc_AirplaneCrash_4j', 8040 : 'Aapc_AirplaneCrash_4k'},
        55: {8041: 'Aapc_AirplaneCrash_5a', 8042: 'Aapc_AirplaneCrash_5b', 8043: 'Aapc_AirplaneCrash_5c', 8044: 'Aapc_AirplaneCrash_5d', 8045: 'Aapc_AirplaneCrash_5e', 8046: 'Aapc_AirplaneCrash_5f', 8047: 'Aapc_AirplaneCrash_5g', 8048: 'Aapc_AirplaneCrash_5h', 8049 : 'Aapc_AirplaneCrash_5i'},
        56: {8050: 'Aapc_AirplaneCrash_6a2', 8051: 'Aapc_AirplaneCrash_6a3', 8052: 'Aapc_AirplaneCrash_6b', 8053: 'Aapc_AirplaneCrash_6c', 8054: 'Aapc_AirplaneCrash_6d2', 8055: 'Aapc_AirplaneCrash_6d3', 8056: 'Aapc_AirplaneCrash_6e', 8057: 'Aapc_AirplaneCrash_6f', 8058: 'Aapc_AirplaneCrash_6g', 8059: 'Aapc_AirplaneCrash_6h', 8060: 'Aapc_AirplaneCrash_6i', 8061: 'Aapc_AirplaneCrash_6j', 8062 : 'Aapc_AirplaneCrash_6k'},
        57: {8063: 'Aapc_AirplaneCrash_7a2', 8064: 'Aapc_AirplaneCrash_7a3', 8065: 'Aapc_AirplaneCrash_7b', 8066: 'Aapc_AirplaneCrash_7c', 8067: 'Aapc_AirplaneCrash_7d2', 8068: 'Aapc_AirplaneCrash_7d3', 8069: 'Aapc_AirplaneCrash_7e', 8070: 'Aapc_AirplaneCrash_7f', 8071: 'Aapc_AirplaneCrash_7g', 8072: 'Aapc_AirplaneCrash_7h', 8073: 'Aapc_AirplaneCrash_7i', 8074: 'Aapc_AirplaneCrash_7j', 8075 : 'Aapc_AirplaneCrash_7k'},
        58: {8076: 'Aapc_AirplaneCrash_8a2', 8077: 'Aapc_AirplaneCrash_8a3', 8078: 'Aapc_AirplaneCrash_8b', 8079: 'Aapc_AirplaneCrash_8c', 8080: 'Aapc_AirplaneCrash_8d3', 8081: 'Aapc_AirplaneCrash_8e', 8082: 'Aapc_AirplaneCrash_8f', 8083: 'Aapc_AirplaneCrash_8g', 8084: 'Aapc_AirplaneCrash_8h', 8085: 'Aapc_AirplaneCrash_8i', 8086: 'Aapc_AirplaneCrash_8j', 8087: 'Aapc_AirplaneCrash_8k', 8088 : 'Aapc_AirplaneCrash_8l'},
        59: {8089: 'Aapc_AirplaneCrash_9a', 8090: 'Aapc_AirplaneCrash_9aa', 8091: 'Aapc_AirplaneCrash_9ab', 8092: 'Aapc_AirplaneCrash_9ac', 8093: 'Aapc_AirplaneCrash_9ad', 8094: 'Aapc_AirplaneCrash_9ae', 8095: 'Aapc_AirplaneCrash_9af', 8096: 'Aapc_AirplaneCrash_9ag', 8097: 'Aapc_AirplaneCrash_9ah', 8098: 'Aapc_AirplaneCrash_9ai', 8099: 'Aapc_AirplaneCrash_9aj', 8100: 'Aapc_AirplaneCrash_9ak', 8101: 'Aapc_AirplaneCrash_9al', 8102: 'Aapc_AirplaneCrash_9am', 8103: 'Aapc_AirplaneCrash_9an', 8104: 'Aapc_AirplaneCrash_9ao', 8105: 'Aapc_AirplaneCrash_9b', 8106: 'Aapc_AirplaneCrash_9c', 8107: 'Aapc_AirplaneCrash_9d', 8108: 'Aapc_AirplaneCrash_9e', 8109: 'Aapc_AirplaneCrash_9f', 8110: 'Aapc_AirplaneCrash_9g', 8111: 'Aapc_AirplaneCrash_9h', 8112: 'Aapc_AirplaneCrash_9i', 8113: 'Aapc_AirplaneCrash_9j', 8114: 'Aapc_AirplaneCrash_9k', 8115: 'Aapc_AirplaneCrash_9l', 8116: 'Aapc_AirplaneCrash_9m', 8117: 'Aapc_AirplaneCrash_9n', 8118: 'Aapc_AirplaneCrash_9o', 8119: 'Aapc_AirplaneCrash_9p', 8120: 'Aapc_AirplaneCrash_9q', 8121: 'Aapc_AirplaneCrash_9r', 8122: 'Aapc_AirplaneCrash_9s', 8123: 'Aapc_AirplaneCrash_9t', 8124: 'Aapc_AirplaneCrash_9u', 8125: 'Aapc_AirplaneCrash_9v', 8126: 'Aapc_AirplaneCrash_9w', 8127: 'Aapc_AirplaneCrash_9x', 8128: 'Aapc_AirplaneCrash_9y', 8129 : 'Aapc_AirplaneCrash_9z'},
        60: {8130: 'Aapc_AirplaneCrash_10a', 8131: 'Aapc_AirplaneCrash_10b', 8132: 'Aapc_AirplaneCrash_10c', 8133: 'Aapc_AirplaneCrash_10d', 8134: 'Aapc_AirplaneCrash_10e', 8135: 'Aapc_AirplaneCrash_10f', 8136: 'Aapc_AirplaneCrash_10g', 8137: 'Aapc_AirplaneCrash_10h', 8138: 'Aapc_AirplaneCrash_10i', 8139: 'Aapc_AirplaneCrash_10j', 8140: 'Aapc_AirplaneCrash_10k', 8141: 'Aapc_AirplaneCrash_10l', 8142 : 'Aapc_AirplaneCrash_10m'},
        61: 'ISS',
        62: 'WDS',
        63: 'Corridor_Tritium',
        64: 'General_Tritium',
        65: 'HDT1',
        66: 'HDT2',
        67: 'HDT3',
        68: 'HDT4',
        69: 'HDT5',
        70: 'HDT6',
        71: 'HDT7',
        72: 'HDT8',
        73: 'HDT9',
        91: 'NBCELLdust_p',
        92: 'NBCELLdust_t',
        501: 'CRLOVAIII',
            }
elif sheetnumber == 3 or sheetnumber == 4:
    LCName = {
        1: 'G+0.5Shr',
        6: 'T_W',
        7: 'Q0_normal_variable_load',
        3100: 'Correction_machine_l3100',
        1001: 'cas_uniX',
        2001: 'cas_uniY',
        3001: 'cas_uniZ',
        4000: 'spectrum_cqc_dep_SL1_x',
        5000: 'spectrum_cqc_dep_SL1_y',
        6000: 'spectrum_cqc_dep_SL1_z',
    }
elif 5 <= sheetnumber <=14 or sheetnumber == 18:
    LCName = {
        1: 'G+0.5Shr',
        2: 'Ge',
        4: 'Tqp_W',
        6: 'T_W',
        7: 'Q0_normal_variable_load',
        8: 'Qe',
        11: 'Crane_load_11',
        12: 'Crane_load_12',
        13: 'Crane_load_13',
        3100: 'Correction_machine_l3100',
        3110: 'Correction_machine_l3110',
        116: 'LC116',
        132: 'LC132',
        266: 'LC266',
        282: 'LC282',
        416: 'LC416',
        432: 'LC432',
        566: 'LC566',
        582: 'LC582',
        839: 'LC839',
        901: 'LC901',
        30: 'GALL2_p',
        402: 'HIG_TK',
        733: 'LC733',
        1176: 'LC1176',
        1244: 'LC1244',
        1178: 'LC1178',
        1246: 'LC1246',
        117: 'LC117',
        133: 'LC133',
        267: 'LC267',
        283: 'LC283',
        417: 'LC417',
        433: 'LC433',
        567: 'LC567',
        583: 'LC583',
        4022: 'HIG_TK_b',
        302: 'GALL2_p_b',
        322: 'DT_p_b',
        1001: 'cas_uniX',
        2001: 'cas_uniY',
        3001: 'cas_uniZ',
        1000: 'spectrum_cqc_dep_SL2_x',
        2000: 'spectrum_cqc_dep_SL2_y',
        3000: 'spectrum_cqc_dep_SL2_z',
        4000: 'spectrum_cqc_dep_SL1_x',
        5000: 'spectrum_cqc_dep_SL1_y',
        6000: 'spectrum_cqc_dep_SL1_z',
        7000: 'spectrum_cqc_dep_SL3_x',
        8000: 'spectrum_cqc_dep_SL3_y',
        9000: 'spectrum_cqc_dep_SL3_z',
        4001: 'Crane1_EX',
        5001: 'Crane1_EY',
        6001: 'Crane1_EZ',
        4002: 'Crane2_EX',
        5002: 'Crane2_EY',
        6002: 'Crane2_EZ',
        4003: 'Crane3_EX',
        5003: 'Crane3_EY',
        6003: 'Crane3_EZ',
    }
elif 15 <= sheetnumber <= 17:
    LCName = {
        1: 'G+0.5Shr',
        4: 'Tqp_W',
        6: 'T_W',
        7: 'Q0_normal_variable_load',
        36: 'CSR_t',
        608: 'LC608',
        650: 'LC650',
        616: 'LC616',
        658: 'LC658',
        630: 'LC630',
        672: 'LC672',
        636: 'LC636',
        678: 'LC678',
        961: 'LC961',
        988: 'LC988',
        970: 'LC970',
        997: 'LC997',
        960: 'LC960',
        987: 'LC987',
        969: 'LC969',
        996: 'LC996',
        501: 'CRLOVAIII',
        1071: 'LC1071',
        1110: 'LC1110',
        1084: 'LC1084',
        1123: 'LC1123',
        1411: 'LC1411',
        1450: 'LC1450',
        1424: 'LC1424',
        1463: 'LC1463',
        1072: 'LC1072',
        1111: 'LC1111',
        1085: 'LC1085',
        1124: 'LC1124',
        1412: 'LC1412',
        1451: 'LC1451',
        1425: 'LC1425',
        1464: 'LC1464',
            }

#############
# Name and label columns, for input and output
columns = {
    "ElemNo": {'type': 'I', 'unit': ''},  
    "cas": {'type': 'I', 'unit': ''},  
    "N_OR": {'type': 'd', 'unit': 'N'},
    "TY_OR": {'type': 'd', 'unit': 'N'},
    "TZ_OR": {'type': 'd', 'unit': 'N'},
    "N_EX": {'type': 'd', 'unit': 'N'},
    "TY_EX": {'type': 'd', 'unit': 'N'},
    "TZ_EX": {'type': 'd', 'unit': 'N'},
    "TORS_OR": {'type': 'd', 'unit': 'N.m'},
    "MZ_OR": {'type': 'd', 'unit': 'N.m'},
    "MY_OR": {'type': 'd', 'unit': 'N.m'},
    "TORS_EX": {'type': 'd', 'unit': 'N.m'},
    "MZ_EX": {'type': 'd', 'unit': 'N.m'},
    "MY_EX": {'type': 'd', 'unit': 'N.m'},
    "A": {'type': 'd', 'unit': 'm²'},
    "IZ": {'type': 'd', 'unit': 'm4'},
    "IY": {'type': 'd', 'unit': 'm4'},
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
for key,name in LCName.items():
    if isinstance(name,dict):
        for k,v in name.items():
            print('Reading load case {}: {}'.format(k, v))
            filePath = os.path.join(path_input, 'beam_' + v + '.csv')
            LC[k] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                       'filePath': filePath,
                       'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime(
                           "%m/%d/%Y %H:%M:%S")}
    else:
        print('Reading load case {}: {}'.format(key, name))
        seismic = {1000: 1001, 2000: 2001, 3000: 3001, 4000: 1001, 5000: 2001, 6000: 3001, 7000: 1001, 8000: 2001, 9000: 3001, }
        sl_reduction = {1000: 1, 2000: 1, 3000: 1, 4000: (1/3), 5000: (1/3), 6000: (1/3), 7000: 1, 8000: 1, 9000: 1, }
        sl1 = {4000: 'spectrum_cqc_dep_SL2_x', 5000: 'spectrum_cqc_dep_SL2_y', 6000: 'spectrum_cqc_dep_SL2_z', }
        sl = {1000: 'spectrum_cqc_dep_SL2_x', 2000: 'spectrum_cqc_dep_SL2_y', 3000: 'spectrum_cqc_dep_SL2_z',
              7000: 'spectrum_cqc_dep_SL3_x', 8000: 'spectrum_cqc_dep_SL3_y', 9000: 'spectrum_cqc_dep_SL3_z'
              }
        if key in sl1.keys():
            filePath = os.path.join(path_input,'beam_' + sl1[key] + '.csv')
            LC[key] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                   'filePath':filePath,
                   'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime("%m/%d/%Y %H:%M:%S")}
            for item in ['N_OR', 'TY_OR', 'TZ_OR', 'N_EX', 'TY_EX', 'TZ_EX', 'TORS_OR', 'MZ_OR', 'MY_OR', 'TORS_EX', 'MZ_EX', 'MY_EX']: #, 'SX_top', 'SX_bot', 'SY_top', 'SY_bot']:
                LC[key]['data'][item] = LC[key]['data'][item] * sl_reduction[key] * np.sign(LC[seismic[key]]['data'][item])
        elif key in sl.keys():
            filePath = os.path.join(path_input,'beam_' + sl[key] + '.csv')
            LC[key] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                   'filePath':filePath,
                   'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime("%m/%d/%Y %H:%M:%S")}
            for item in ['N_OR', 'TY_OR', 'TZ_OR', 'N_EX', 'TY_EX', 'TZ_EX', 'TORS_OR', 'MZ_OR', 'MY_OR', 'TORS_EX', 'MZ_EX', 'MY_EX']: #, 'SX_top', 'SX_bot', 'SY_top', 'SY_bot']:
                LC[key]['data'][item] = LC[key]['data'][item] * np.sign(LC[seismic[key]]['data'][item])
        else:
            filePath = os.path.join(path_input, 'beam_' + name + '.csv')
            LC[key] = {'data': np.genfromtxt(filePath, delimiter=',', skip_header=9, dtype=dtypecsv),
                       'filePath': filePath,
                       'modifiedDate': datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).strftime(
                           "%m/%d/%Y %H:%M:%S")}
sheet = workbook
columns = sheet.columns.values
size = sheet.shape


for row in range(1,size[0]):
# for row in range(177,size[0]):

    # Dimension a large array for output - More efficient to preallocate in numpy
    COMB = np.empty(shape=(LC[1]['data'].size,), dtype=dtypecsv)

    LCsUsed = []
    comboStrs = ['! ','! ','! ']
    for col in columns[6:size[1]]:
        if sheet[col][row] is '':
            continue
        if col == 'END':
            break
        if col != 'Load case Num.':
            LCsUsed.append(col)

            #Create strings to show the combinations - Useful for debugging
            if comboStrs[0] != '! ':
                for j in [0,1,2]:
                    comboStrs[j] += '  +  '

            if col in [51, 52, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
                lcfact = 1
                comboStrs[0] += '{factors} x [{comboName}]'.format(factors=str(sheet[col][row]),
                                                                   comboName=col)
                comboStrs[1] += '{factors} x [{comboNum}]'.format(factors=str(sheet[col][row]),
                                                                  comboNum=sheet[col][0])
                comboStrs[2] += '{evalFactor:.3f} x [{comboNum}]'.format(comboNum=col, evalFactor=lcfact)

                for item in ['N_OR', 'TY_OR', 'TZ_OR', 'N_EX', 'TY_EX', 'TZ_EX', 'TORS_OR', 'MZ_OR', 'MY_OR', 'TORS_EX', 'MZ_EX', 'MY_EX']:
                    COMB[item] += LC[sheet['Load case Num.'][row]]['data'][item]*lcfact
                for item in ['ElemNo', 'A', 'IZ', 'IY']:
                    COMB[item] = LC[sheet['Load case Num.'][row]]['data'][item]
                for item in ['cas']:
                    COMB[item] = 0 #np.full((len(LC[1]),), rows2[14].value)
            else:
                lcfact = eval(str(sheet[col][row]))
                comboStrs[0] += '{factors} x [{comboName}]'.format(factors=str(sheet[col][row]),
                                                                   comboName=col)  # LCName[col[i]]
                comboStrs[1] += '{factors} x [{comboNum}]'.format(factors=str(sheet[col][row]),
                                                                      comboNum=sheet[col][0])
                comboStrs[2] += '{evalFactor:.3f} x [{comboNum}]'.format(comboNum=col, evalFactor=lcfact)

            for item in ['N_OR', 'TY_OR', 'TZ_OR', 'N_EX', 'TY_EX', 'TZ_EX', 'TORS_OR', 'MZ_OR', 'MY_OR','TORS_EX','MZ_EX','MY_EX']:
                COMB[item] += LC[col]['data'][item]*lcfact
            for item in ['ElemNo', 'A', 'IZ', 'IY']:
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
                    path_output,worksheet_names[sheetnumber],'beam_combin' + "{:07d}".format(sheet['Load case Num.'][row]) + '.csv'), "w") as f:

        print('Writing file: {}'.format(f.name))
        #Create headers
        f.write(comboStr + "\n")
        f.write("!" + "\n")
        for LCNum in LCsUsed:
            if LCNum in [51, 52, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
                f.write('! [{LCNum}], {date}, {filePath}\n'.format(LCNum=sheet['Load case Num.'][row], date=LC[sheet['Load case Num.'][row]]['modifiedDate'], fmt=fmt, filePath=LC[sheet['Load case Num.'][row]]['filePath']))
            elif LCNum in [4000,5000,6000]:
                f.write('! [{LCNum}], {date}, {filePath} !!(SL1 = 1/3*SL2) \n'.format(LCNum=LCNum, date=LC[LCNum]['modifiedDate'], fmt=fmt, filePath=LC[LCNum]['filePath']))
            else:
                f.write('! [{LCNum}], {date}, {filePath}\n'.format(LCNum=LCNum, date=LC[LCNum]['modifiedDate'], fmt=fmt, filePath=LC[LCNum]['filePath']))

        f.write("!" + "\n")
        for header in [columnNames,columnUnits]:
            headerStr = '! '
            for colValue in header:
                headerStr += colValue + ','
            f.write(headerStr + "\n")

        #Output data using previous definitions
        np.savetxt(f, COMB, fmt=fmt) #delimiter=','

