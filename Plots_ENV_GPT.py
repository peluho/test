import os  # importing os library
import glob  # importing glob library
import re
import time, datetime
import math
import fnmatch
import sys
import os

# get current date and time
now = datetime.datetime.now()
# format the date and time as a string
timestamp = now.strftime("%Y-%m-%d")

time_start = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
user = os.getlogin()

# input_folder = '20230327_103316'
input_folder = '20230316_181757'
# input_folder = '20230214_161427'
path_input = '\\\\io-ws-ccstore1\\03.Ansys\\04_pythonScripts\\02_Landa_N_Plot\\09_Output\\' + input_folder +'\\'
pathDB = '\\\\io-ws-ccstore1\\03.Ansys\\02_5F\\01_Config1\\01_DB\\'
path_out = '\\\\io-ws-ccstore1\\03.Ansys\\04_pythonScripts\\02_Landa_N_Plot\\09_Output\\'
path_Macros = '\\\\io-ws-ccstore1\\03.Ansys\\04_pythonScripts\\02_Landa_N_Plot\\09_Output\\new_plots\\'
if not os.path.exists(os.path.join(path_out,'new_plots')):
    os.makedirs(os.path.join(path_out,'new_plots'))

def list_files_with_extension(directory, extension):
    """
    Given a directory path and an extension, this function returns a list of all file names in the directory
    that have the specified extension.
    """
    files_with_extension = []
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            files_with_extension.append(file_name)
    return files_with_extension

def find_files_with_pattern(pattern, directory):
    """
    Given a pattern and a directory path, this function returns a list of all file names in the directory
    that match the pattern.
    """
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches
def split_path_and_filename(file_paths):
    """
    Given a list of file paths, this function splits each path into a directory and filename and returns two lists -
    one containing the directories and the other containing the filenames.
    """
    directories = []
    filenames = []
    for file_path in file_paths:
        directory, filename = os.path.split(file_path)
        directories.append(directory)
        filenames.append(filename)
    return directories, filenames

def list_directories_and_filenames(file_paths):
    """
    Given a list of file paths, this function splits each path into a directory and filename and returns a dictionary of
    lists, where the keys are the structural elements and the values are dictionaries containing the old and new filenames.
    The new filenames are modified to replace certain patterns.
    """
    directories_by_element = {element: [] for element in ["LB", "B2", "BIO", "CROWN"]}
    files_by_element = {element: [] for element in ["LB", "B2", "BIO", "CROWN"]}
    for file_path in file_paths:
        directory, filename = os.path.split(file_path)
        for element in directories_by_element.keys():
            if element in filename:
                directories_by_element[element].append(directory)
                old_filename = filename
                new_filename = old_filename.replace('_Rh46Results', '').replace('M1M2-M1M2', 'M1M2').replace('CAT_I_II', 'cII') \
                    .replace('CAT_III_IV', 'cIV').replace('CAT_V', 'cV').replace('-SmAvg', '').replace('WS_WOS', 'ENV') \
                    .replace('-5X5F', '')
                new_filename = new_filename.split('.')[0]  # remove extension
                old_filename = old_filename.split('.')[0]  #remove extension
                files_by_element[element].append({'old': old_filename, 'new': new_filename})
                break
    return {
        'directories_by_element': directories_by_element,
        'files_by_element': files_by_element
    }


def extract_keys(d, keys=[]):
    for key, value in d.items():
        keys.append(key)
        if isinstance(value, dict):
            extract_keys(value, keys)
    return list(set(keys))

def write_grid_lines(lineplan, element):
    if element in ['B2', 'LB']:
        with open(os.path.join(path_out, 'new_plots', f'Input_macro_lineplan-11_{element}.inp'), 'w') as f:
            f.write(f'!!!!!! Grid lines generated for {element}\n')
            f.write('/ ANNO, DELE\n')
            f.write('ERASE\n')
            f.write('csys,0 \n')
            f.write('\n')
            f.write('!!!!!Grid Lines from  T1 to T15\n')
            f.write('\n')
            f.write('/AN3D,ANUM,0,102,0,0,0\n')
            f.write('\n')
            for i in range(1, 16):
                x = -57.65 + (i-1) * 7.4
                f.write(f'/AN3D,Line,{x},{lineplan[element]["Ymin"]}-0.4,0,{x},{lineplan[element]["Ymax"]}+0.4,0\n')
                f.write(f'/AN3D,TEXT,{x},{lineplan[element]["Ymax"]}+0.4,0,T{i}\n')
            f.write('\n')
            f.write('!!!!!!Gird Lines from TA to TJ\n')
            f.write('\n')
            for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
                x = lineplan[element]["Xmin"] - 10 + i * 7.4
                f.write(f'/AN3D,Line,{x},-35.5,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,-35.5,0,{letter}A\n')
                f.write(f'/AN3D,Line,{x},-24.6,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,-24.6,0,{letter}B\n')
                f.write(f'/AN3D,Line,{x},-15.8,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,-15.8,0,{letter}C\n')
                f.write(f'/AN3D,Line,{x},-7.9,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,-7.9,0,{letter}D\n')
                f.write(f'/AN3D,Line,{x},0,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,0,0,{letter}E\n')
                f.write(f'/AN3D,Line,{x},7.9,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
                f.write(f'/AN3D,TEXT,{x}-10,7.9,0,{letter}F\n')
                f.write(f'/AN3D,Line,{x},15.8,0,{x},{lineplan[element]["Xmax"]}+10,0\n')
    return

search_term = '*h46*.inp'
inp_files = find_files_with_pattern(search_term, path_input)
result = list_directories_and_filenames(inp_files)
directories_by_element = result['directories_by_element']
# filenames = result['filenames']
files_by_element = result['files_by_element']

# Print the old and new filenames
for element, element_files in files_by_element.items():
    print(f"{element}:")
    for file_dict in element_files:
        print(f"  Old: {file_dict['old']}")
        print(f"  New: {file_dict['new']}")

# Print the old and new directory
for directory, element_directories in directories_by_element.items():
    print(f"{directory}:")
    for directory_dict in element_directories:
        print(f"  Directory: {directory_dict}")

pattern_dict = {}
pattern_dict = {
    'B2': 11,
    'LB': 11,
    'BIO': {
        'Bioshield_SE': 22,
        'Bioshield_EN': 20,
        'Bioshield_NW': 19,
        'Bioshield_WS': 23
    }}
pattern_dict['CROWN'] = {}
for i in [0, 80, 160, 280]:
    pattern_dict['CROWN'][f'Short_CW_{i}_{i+20}'] = [i, i+20, 'Short', 'SCW']
pattern_dict['CROWN']['Long_CW_40_60'] = [40, 60, 'Long', 'LCW']
pattern_dict['CROWN']['Long_CW_320_340'] = [320, 340, 'Long', 'LCW']
for i in range(0, 341, 20):
    pattern_dict['CROWN'][f'RW_{i}'] = [i]

lineplan_bounds = [    ('B2', -55, 55, -45, 50, -12.35, -12.35),    ('LB', -55, 55, -45, 50, -13.97, -13.97),    ('Bioshield_SE', 0, 15.235, -15.004, 0, -12.35, -5.5),    ('Bioshield_EN', 0, 15.235, 0, 15.004, -12.35, -5.5),    ('Bioshield_NW', -15.235, 0, 0, 15.004, -12.35, -5.5),    ('Bioshield_WS', -15.235, 0, -15.004, 0, -12.35, -5.5),    ('Short_CW_0_20', 0, 15.235, 0, 15.004, -12.35, -5.5),    ('Short_CW_80_100', -15, 15, 0, 0, -12.35, -5.5),    ('Short_CW_160_180', -15.235, 0, 0, 15.004, -12.35, -5.5),    ('Short_CW_280_300', 0, 15.235, -15.004, 0, -12.35, -5.5),    ('Long_CW_40_60', 0, 15.235, 0, 15.004, -12.35, -5.5),    ('Long_CW_320_340', 0, 15.235, -15.004, 0, -12.35, -5.5)]

lineplan_dict = {item[0]: {'Xmin': item[1], 'Xmax': item[2], 'Ymin': item[3], 'Ymax': item[4], 'Zmin': item[5], 'Zmax': item[6]} for item in lineplan_bounds}

CW_view_dict = {
    'Short_CW_0_20': {
        'view_XV': 170.0,
        'view_YV': -10.0,
        'focus_XF': 4.31504062816,
        'focus_YF': 0.825090282390
    },
    'Short_CW_80_100': {
        'view_XV': 90.0,
        'view_YV': -90.0,
        'focus_XF': -0.726625362900E-01,
        'focus_YF': 2.68820594049
    },
    'Short_CW_160_180': {
        'view_XV': 10.0,
        'view_YV': -10.0,
        'focus_XF': -0.549202160984,
        'focus_YF': -0.151133129637E-01
    },
    'Short_CW_280_300': {
        'view_XV': 110.0,
        'view_YV': 70.0,
        'focus_XF': 0.127381808570,
        'focus_YF': 0.231126675956
    },
    'Long_CW_40_60': {
        'view_XV': 130.0,
        'view_YV': -50.0,
        'focus_XF': 2.09483746371,
        'focus_YF': 2.68820594049
    },
    'Long_CW_320_340': {
        'view_XV': 150.0,
        'view_YV': 30.0,
        'focus_XF': 0.748806535319E-01,
        'focus_YF': 0.140194675430
    }
}

etables = ['AXS','AXI','AYS','AYI','ATR','AXSUF','AXIUF','AYSUF','AYIUF','ATRUF','Landa_N']


keys_to_exclude = ['CROWN', 'BIO']
keys = list(pattern_dict.keys())
keys = [k for k in keys if k not in keys_to_exclude]

bio_keys = []
if 'BIO' in pattern_dict:
    bio_keys = [k for k in pattern_dict['BIO'].keys() if k not in keys_to_exclude]
keys += bio_keys

cw_keys = ['Short_CW_0_20', 'Short_CW_80_100', 'Short_CW_160_180', 'Short_CW_280_300', 'Long_CW_40_60', 'Long_CW_320_340']

rw_keys = [f'RW_{i}' for i in range(0, 341, 20)]

crown_keys =[]
crown_keys = cw_keys + rw_keys

keys += cw_keys + rw_keys

keys.sort()

#remove the empty elements
files_by_element = {k: v for k, v in files_by_element.items() if v}
directories_by_element = {k: v for k, v in directories_by_element.items() if v}

elements = []
elements = files_by_element.keys()
# elements = ['LB','B2','BIO','CROWN']

for element in elements:
    # create an empty dictionary
    actions = {}

    common_part = [
        "fini $ /cle",
        f"resume,Tokamak_Model_Reinforcement,db,{pathDB}   !Any DB is working",
        "/post1",
        "*get,twall,active,,time,wall  !saves the wall time before solution",
        "*get,tcpubefore,active,,time,CPU !saves the CPU time before solution",
        f"! Path were *.inp were extracted from:' + {str(path_input)}",
        "",
        f"! Components of the structural element to be plotted:  {element}",
    ]

    actions["LB"] = common_part + [
        "!!Selection of the lower basemat - LB slab",
        "csys, 1",
        "esel,,real,,51",
        "esel,r,ename,,43",
        "!esel, r, cent, x, 0, 31",
        "csys, 0",
        "cm,LB,elem",
        "",
        "!!Selection of the lower basemat for Configuration 2- LB slab",
        "csys, 1",
        "esel,,real,,51",
        "esel,r,ename,,43",
        "esel, r, cent, x, 0, 31",
        "csys, 0",
        "cm,LB_C2,elem",
        "usePartialResults=1",
        "",
        "!!Selection of the lower basemat SHEAR - Shear LB slab",
        "esel,s,cent,z,-13.97,-12.35",
        "esel,r,type,,1",
        "nsle",
        "nsel,r,loc,z,-13.97",
        "nplot",
        "nsle",
        "esln",
        "nsle",
        "esln",
        "cm,shear_LB,elem",
    ]
    # Elements used in the B2 shear selection
    codes = [18253, 18369, 18378, 18386, 18388, 18390, 18393, 18396, 18487, 18611, 18747, 18756, 18757, 18763, 20028,
             20664, 20666, 20667, 20672, 20685, 20787, 20790, 20796, 20805, 20827, 21238, 21240, 21289, 21299, 21304,
             21516, 21523, 22182, 22538, 22542, 22543, 22546, 22550, 22554, 22555, 22560, 22634, 23090, 23091, 23141,
             23142, 23143, 23201, 23207, 23212, 23293, 23309, 23455, 23493, 23499, 23505, 23520, 23521, 23529, 23546,
             23677, 25643, 26237]
    # loop through the codes and generate the "esel" commands
    B2_esel_part = [f"esel,a,elem,{code}" for code in codes]
    B2_continue = [
        "cm,SHEAR_1,elem",
        "esel,r,ename,,43",
        "cmsel,s,SHEAR_1",
        "csys,1",
        "esel,r,cent,x,14,8",
        "nsle",
        "esln",
        "nsle",
        "esln",
        "cm,SHEAR_2,elem",
        "cmsel,s,SHEAR_1",
        "cmsel,a,SHEAR_2",
        f"cm,shear_{element},elem",
    ]

    actions["B2"] = common_part + [
                       "!!Selection of the upper basemat - B2 slab",
                       "csys, 1",
                       "esel,, cent, z, -12.35",
                       "esel,u,mat,,123,126,3",
                       "esel,u,real,,112,113,1",
                       "esel, r, ename,, 43",
                       "csys, 0",
                       "cm,B2,elem",
                       "usePartialResults=1",
                       "",
                       "!!Selection of the upper basemat for config 2- B2 slab",
                       "csys, 1",
                       "esel,, cent, z, -12.35",
                       "esel, r, cent, x, 0, 31",
                       "esel, r, ename,, 43",
                       "csys, 0",
                       "cm,B2_C2,elem",
                       "usePartialResults=1",
                       "",
                       "!!Selection of the lower basemat SHEAR - Shear LB slab",
                       "csys,0",
                       "esel,s,cent,z,-12.30,-11",
                       "nsle",
                       "nsel,r,loc,z,-12.35,12.1",
                       "nsle",
                       "esln",
                       "nsle",
                       "esel,r,cent,z,-12.35",
                       "cm,B2_1,elem",
                       "esel,s,cent,z,-12.35",
                       "esel,r,cent,x,34,36.5",
                       "nsle",
                       "cm,B2_2,elem",
                       "cmsel,s,B2_1",
                       "cmsel,a,B2_2",
                       "cm,B2_0,elem",
                       "esel,s,type,,12",
                       "cm,ASB_ALL,elem",
                       "nsle",
                       "esln",
                       "esel,r,cent,z,-12.35",
                       "nsle",
                       "cm,LB,elem",
                       "cmsel,s,B2_0",
                       "cmsel,a,LB",
                       "cm,B2_ALL_1,elem",
                       "!Columns refinement",
                       "cmsel,s,B2_ALL_1",
                       "esel,r,cent,x,-53,-40",
                       "esel,r,cent,y,-18,25.5",
                       "nsle",
                       "esln",
                       "cm,B2_3,elem",
                       "!inner circ wall refinement",
                       "!cylindrics coords.",
                       "csys,1",
                       "esel,s,cent,z,-12.35",
                       "cmsel,s,B2_ALL_1",
                       "esel,r,cent,x,23.5,26",
                       "nsle",
                       "esln",
                       "cm,B2_4,elem",
                       "cmsel,s,B2_ALL_1",
                       "cmsel,a,B2_3",
                       "cmsel,a,B2_4",
                       "cm,B2_ALL_2,elem",
                       "!ele by ele refinement",
                       "cmsel,s,B2_all_2",
                   ] + B2_esel_part + B2_continue

    walls = {
        'Bioshield_SE': {'X': [-0.1, 15.335], 'Y': [-15.104, 0.1]},
        'Bioshield_EN': {'X': [-0.1, 15.335], 'Y': [-0.1, 15.104]},
        'Bioshield_NW': {'X': [-15.335, 0.1], 'Y': [-0.1, 15.104]},
        'Bioshield_WS': {'X': [-15.335, 0.1], 'Y': [-15.104, 0.1]},
    }

    actions['BIO'] = []
    actions['BIO'] += common_part
    for wall_name, coords in walls.items():
        x_coords, y_coords = coords['X'], coords['Y']
        selection = (
            f"!!Selection of the Bioshield - Wall {wall_name}\n"
            f"NSEL, S, LOC, X, {x_coords[0]}, {x_coords[1]}\n"
            f"NSEL, R, LOC, Y, {y_coords[0]}, {y_coords[1]}\n"
            "NSEL, R, LOC, Z, -12.4, -5.45\n"
            "ESLN, S, 1, ALL\n"
            "esel, r, real,, 160\n"
            "ESEL, R, TYPE,, 7\n"
            "nsle, s, 1\n"
            f"cm,{wall_name},elem\n"
            "usePartialResults=1\n"
            "\n"
        )
        actions['BIO'].append(selection)

    actions['CROWN'] = common_part + [
        "!! Components of the structural element to be plotted: CROWN - Short and Long Circular Walls", "",
        "!! All circular walls", "", "csys,1", "allsel", "esel,s,real,,157", "esel,u,esys,,88,128,40",
        "esel,r,cent,x,0,10.68", "", "cm,short_circular_walls_all,elem", "!---------------------",
        "!Elements representing steel transition piece (STP) are not considered",
        "!(see section 6.3 of ENG-51-CR-110011-CW-v03.0)", "", "alls", "cmsel,s,short_circular_walls_all,elem", "eplot",
        "nsle,r,all", ]

    for i, (y1, y2, z1, z2) in enumerate(
            [(159, 165, -9.39, -7.98), (176, 180, -9.39, -7.98), (79, 85,-9.38,-7.98), (95.5, 101,-9.38,-7.98),
             (0, 4.3,-9.38,-7.98), (15.6, 21,-9.38,-7.98), (-81, -75.3,-9.37,-7.98), (-64.5, -59,-9.37,-7.98)]):
        f = f"STP_{i + 1}"
        actions['CROWN'] += [
            f"nsel,r,loc,y,{y1},{y2}",
            f"nsel,r,loc,z,{z1},{z2}",
            "esln,r,1",
            f"cm,{f},elem",
            "",
            "alls",
            f"cmsel,s,short_circular_walls_all,elem",
            "eplot",
            "nsle,r,all",
        ]

    # Create a list of strings
    actions['CROWN'] += ['',
               'alls',
               '!Removed elements',
               'esel,s,sec,,130',
               'esel,a,sec,,131',
               'esel,a,sec,,132',
               'nsle',
               'nsel,u,,,23469',
               'esln',
               'esel,r,type,,7',
               'cm,STP_elem,elem',
               '',
               '',
               '',
               'alls',
               'cmsel,s,STP_1,elem',
               'cmsel,a,STP_2,elem',
               'cmsel,a,STP_3,elem',
               'cmsel,a,STP_4,elem',
               'cmsel,a,STP_5,elem',
               'cmsel,a,STP_6,elem',
               'cmsel,a,STP_7,elem',
               'cmsel,a,STP_8,elem',
               'cm,STP_short,elem',
               '',
               'alls',
               'cmsel,s,short_circular_walls_all,elem']

    # Append formatted strings to actions list
    for i in range(1, 9):
        actions['CROWN'].append(f"cmsel,u,STP_{i},elem")

    actions['CROWN'] += [f"cm,short_circular_walls,elem",
                '',
                '!------------------------------------------------------',
                '!Definition of components',
                '!------------------------------------------------------',
                '',
                '!Radial Walls',
                '',
                'allsel',
                'esel,s,real,,157',
                'esel,r,esys,,88,128,40',
                '',
                'cm,circular_walls_all,elem',
                '',
                '!Elements representing steel transition piece (STP) are not considered',
                '!(see section 6.3 of ENG-51-CR-110011-CW-v03.0)',
                '',
                'csys,1',
                'eplot',
                'nsle,r,all']


    # Append formatted strings to actions list
    for i, (y1, y2, z1, z2) in enumerate([(39, 45,-9.36,-7.98), (55, 60.1,-9.36,-7.98),
                                          (-24.3, 20,-9.37,-7.98), (-40, -35.4,-9.38,-7.98)]):
        actions['CROWN'] += [
        f"nsel,r,loc,y,{y1},{y2}",
        f"nsel,r,loc,z,{z1},{z2}",
        'esln,r,1',
        f'cm,STP_{i+1},elem',
        "",
        'alls',
        'cmsel,s,circular_walls_all,elem',
        'eplot',
        'nsle,r,all',
        ]
    # Append formatted strings to actions list
    actions['CROWN'] += ['alls',
                'cmsel,s,circular_walls_all,elem',
                'cmsel,u,STP_1,elem',
                'cmsel,u,STP_2,elem',
                'cmsel,u,STP_3,elem',
                'cmsel,u,STP_4,elem',
                'cm,circular_walls,elem',
                '',
                'alls',
                'cmsel,s,STP_1,elem',
                'cmsel,a,STP_2,elem',
                'cmsel,a,STP_3,elem',
                'cmsel,a,STP_4,elem',
                'cm,STP_long,elem',
                '',
                '!----------------------------------------------',
                '!----------------------------------------',
                'alls,',
                'cmsel,s,STP_short,elem',
                'cmsel,a,STP_long,elem',
                'cm,STP_elem,elem',
                'alls']

    for cw_key in cw_keys:
        crown = pattern_dict['CROWN'][cw_key]
        actions['CROWN'] += [
            f"!!Selection of the {crown[2]} circular wall - {crown[0]} to {crown[1]}",
            "csys,1",
            "esel,,real,,157",
            "nsle",
            "nsel,,loc,x,10.2,10.7",
            "nsel,r,loc,z,-12.36,-7.9",
            f"nsel,r,loc,y,{crown[0]}-2,{crown[1]}+2",
            "esln,r,1",
            "cmsel,u,STP_elem",
            "dsys,0",
            f"cm,{cw_key},elem"
        ]

    actions['CROWN'] += [
        f"!------------------------------------------------------",
        f"!Definition of components",
        f"!------------------------------------------------------",
        f"",
        f"!Radial Walls",
        f"",
        f"allsel",
        f"csys,1",
        f"esel,s,real,,155",
        f"esel,a,real,,162",
        f"esel,r,cent,x,0,20",
        f"cm,radial_walls_all,elem",
        f"",
        f"!Elements representing steel transition piece are not considered",
        f"!(see section 6.3 of ENG-51-CR-110011-CW-v03.0)",
        f"",
        f"eplot",
        f"nsle,r,alls",
        f"nsel,r,loc,x,0,11.90",
        f"nsel,r,loc,z,-9.31,-7.98",
        f"esln,r,1",
        f"",
        f"cm,steel_transition_piece,elem",
        f"",
        f"alls",
        f"cmsel,s,radial_walls_all,elem",
        f"cmsel,u,steel_transition_piece,elem",
        f"cm,radial_walls,elem",
    ]

    # Open the file with the modified stru_elem string
    f = open(os.path.join(path_out, 'new_plots', f'Plot_{element}_{timestamp}.inp'), 'w')

    if element in actions:
        view = pattern_dict[element]
        for action in actions[element]:
            f.write(action + '\n')
    else:
        # handle case when stru_elem not in actions
        pass

    # close file here

    f.write('\n')
    folder_name = ""
    if 'Bio' in element:
        folder_name = "BIO"
    elif 'CW' in element:
        folder_name = "CROWN\CW"
    elif 'RW' in element:
        folder_name = "CROWN\RW"
    else:
        folder_name = element
    f.write(f'/mkdir,.\{input_folder}\{folder_name}\n')
    f.write(f'/inquire,warn_exist,exist,.\{input_folder}\{folder_name}\warnings,txt,,\n')
    f.write(f'/inquire,warn_date,date,.\{input_folder}\{folder_name}\warnings,txt,,\n')
    f.write(f'\n')
    f.write(f'*if,warn_exist,eq,1,then\n')
    f.write(f'    /rename,.\{input_folder}\{folder_name}\warnings,txt,,.\{input_folder}\{folder_name}\warnings_%warn_date%,txt,,\n')
    f.write(f'*endif\n')
    f.write(f'\n')
    f.write(f'*cfo,.\{input_folder}\{folder_name}\warnings,txt,, \n')
    f.write(f'*vwr \n')
    f.write(f"Warning file generated the {time_start} by {user}\n")
    f.write(f'*vwr \n')
    f.write(f"FILE NAME \t WARNING \t FOLDER \n")
    f.write(f'*cfc \n')
    f.write(f'\n')
    # print('*************DEBUG*************************')
    # print(f"files_by_element keys: {files_by_element.keys()}")
    # print(f'{range(len(files_by_element[element]))}')
    # print('*************DEBUG*************************')
    f.write('*do,ii,1,' + str(len(files_by_element[element])) + ',1' + '\n')
    for i in range(len(files_by_element[element])):
        if i == 0:
            f.write(f"   *if,ii,eq,{i + 1},then\n")
        else:
            f.write(f"   *elseif,ii,eq,{i + 1},then\n")
        # print(type(element_files[i]['new']))
        f.write(f"       name='{files_by_element[element][i]['new']}'\n")
        if 'C2' in files_by_element[element][i]['new']:
            f.write("       conf = 2\n")
        else:
            f.write("       conf = 1\n")
        target_margin = {}
        if 'SL' in files_by_element[element][i]['new'] and 'SL3' not in files_by_element[element][i]['new'] and 'VDE' not in files_by_element[element][i]['new']:
            if 'cII' in files_by_element[element][i]['new']:
                target_margin['SL'] = 1.101  # SL cat I/II
                f.write("       target_margin = 1.101   !SL cat I/II\n")
            elif 'cIV' in files_by_element[element][i]['new']:
                target_margin['SL'] = 1.051  # SL cat III/IV
                f.write("       target_margin = 1.051   !SL cat III/IV\n")
            elif 'cV' in files_by_element[element][i]['new']:
                target_margin['SL'] = 1.052  # SL cat V
                f.write("       target_margin = 1.052   !SL cat V\n")
            else:
                target_margin['SL'] = 1  # default value
        elif 'VDE' in files_by_element[element][i]['new']:
            if 'cII' in files_by_element[element][i]['new']:
                target_margin['VDE'] = 1.138  # VDE cat I/II
                f.write("       target_margin = 1.138   !VDE cat I/II\n")
            elif 'cIV' in files_by_element[element][i]['new']:
                target_margin['VDE'] = 1.108  # VDE cat III/IV
                f.write("       target_margin = 1.108   !VDE cat III/IV\n")
            elif 'cV' in files_by_element[element][i]['new']:
                if 'SL' in files_by_element[element][i]['new']:
                    target_margin['VDE'] = 1.051  # VDE cat V
                    f.write("       target_margin = 1.051   !SL + VDE cat V\n")
                else:
                    target_margin['VDE'] = 1.051  # VDE cat V
                    f.write("       target_margin = 1.051   !VDE cat V\n")
            # else:
            #     target_margin['SL_VDE'] = 1.051  # default value
        else:
            target_margin['default'] = 1  # default value for other cases
            f.write("       target_margin = 1\n")

        f.write('       allsel,all\n')
        f.write(f"       /inquire,filesize,size,{directories_by_element[element][i]}\\{files_by_element[element][i]['old']},inp\n")
        f.write(f"       fkb = filesize*1024\n")
        f.write(f"          *if, fkb, lt, 5, then\n")
        if 'Bio' in element:
            f.write(f'              *cfo,.\{input_folder}\BIO\warnings,txt,,append\n')
        else:
            f.write(f'              *cfo,.\{input_folder}\{element}\warnings,txt,,append\n')

        f.write(f'              *vwr\n')
        f.write(f"              ('{files_by_element[element][i]['old']},inp\tNO RESULTS\t{directories_by_element[element][i]}')\n")
        f.write(f'              *cfc\n')
        for etable in etables:
            f.write(f"              ETABLE,{etable}\n")
        # if fkb < 5:
        f.write(f"          *else\n")
        f.write(f"               /input,'{files_by_element[element][i]['old']}','inp','{directories_by_element[element][i]} '\n")
        f.write(f"          *endif\n")
    f.write(f"    *endif\n")

    write_views = {}

    bio_views = {
        11: {
            'input': 'Input_macro_lineplan-11_',
            'rep': 'FAST',
            'view': '1,0,0,1',
            'vup': ',-x',
            'ang': ['1', '1,90,ZS,1'],
            'foc': ''
        },
        12: {
            'input': 'Input_macro_lineplan-12_',
            'rep': 'FAST',
            'view': '1,0,-1,0',
            'vup': ',,z',
            'ang': ['1'],
            'foc': ''
        },
        13: {
            'input': 'Input_macro_lineplan-13_',
            'rep': 'FAST',
            'view': '1,1',
            'vup': ',,y',
            'ang': ['1', '-90,ZS'],
            'foc': ''
        },
        14: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,1,0.5,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': '1,12.8385670721,3.94929454584,-10.2609282309'
        },
        15: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,0.4,-1,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        16: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,0.5,1,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        17: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,1,-0.66,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        18: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,0,1,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        19: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,-1,1,0',
            'vup': '',
            'ang': ['90,ZS'],
            'foc': ''
        },
        20: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,1,1,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        21: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,-1,0.33,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        22: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,1,-1,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        23: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,-1,-1,0',
            'vup': '',
            'ang': ['90,ZS'],
            'foc': ''
        },
        24: {
            'input': 'Input_macro_lineplan-ang_',
            'rep': 'FAST',
            'view': '1,-1,0,0',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
        "default": {
            'input': '',
            'rep': 'FAST',
            'view': '1,1,-1,1',
            'vup': '',
            'ang': ['-90,ZS'],
            'foc': ''
        },
    }
    #This block set teh view for the different structural elements
    counter = 1
    if "BIO" in element:
        f.write('*do,jj,1,' + str(len(pattern_dict['BIO'])) + ',1 \n')
        for stru_elem in bio_keys:
            view = pattern_dict['BIO'][stru_elem]
            bio_view = bio_views.get(view, bio_views['default'])
            if counter == 1:
                f.write(f'   *if,jj,eq,{counter},then \n')
                f.write(f"        reinf_comp = '{stru_elem}'\n")
            else:
                f.write(f'   *elseif,jj,eq,{counter},then \n')
                f.write(f"        reinf_comp = '{stru_elem}'\n")
            input_str = bio_view['input'] + stru_elem if bio_view['input'] else ''
            f.write(f"        /input,{input_str},inp,'{path_Macros}'\n")
            f.write('        /REP,FAST\n')
            f.write(f'        /VIEW,{bio_view["view"]}\n')
            # f.write(f'        /VUP{bio_view["vup"]}\n')
            for angs in bio_view['ang']:
                f.write(f'        /ANG,1,{angs}\n')
            counter += 1
        f.write('   *endif\n')
        counter = 1
    elif "CROWN" in element:
        f.write('*do,jj,1,' + str(len(pattern_dict['CROWN'])) + ',1 \n')
        for stru_elem in crown_keys:
            if counter == 1:
                f.write('   *if,jj,eq,' + str(counter) + ',then \n')
            else:
                f.write('   *elseif,jj,eq,' + str(counter) + ',then \n')
            if 'CW' in stru_elem:
                f.write("        reinf_comp = '" + stru_elem + "'\n")
                f.write("        !/input,Input_macro_lineplan-11_" + stru_elem + ",inp,'" + path_Macros + "'" + '\n')
                f.write("        cmsel,,%reinf_comp% \n")
                f.write("        nsle \n")
                f.write('        /REP,FAST' + '\n')
                f.write(
                    f"        /VIEW,1,{math.cos(math.radians(CW_view_dict[stru_elem]['view_XV']))},{math.sin(math.radians(CW_view_dict[stru_elem]['view_YV']))},0\n")
                f.write('        /VUP,1,Z' + '\n')
                f.write('        /ANG,1' + '\n')
                f.write(f'        /FOC,1,{CW_view_dict[stru_elem]["focus_XF"]},{CW_view_dict[stru_elem]["focus_YF"]},-10.3134957630' + '\n')
                f.write('        /DIST,1,5' + '\n')
                f.write('        /dscale,1,off' + '\n')
                f.write('        /AUTO,1' + '\n')
                f.write('        /REP,FAST' + '\n')
            elif 'RW' in stru_elem:
                f.write(f"        csys,1 \n")
                f.write(f"        cmsel,s,radial_walls,elem \n")
                f.write(f"        alls,below,elem \n")
                f.write(f"        nsel,r,loc,y,20*({counter-6}-1)-10,20*({counter-6}-1)+10 \n")
                f.write(f"        esln,r,1 \n")
                f.write(f"        cm,{stru_elem},elem \n")
                f.write(f"        reinf_comp = '{stru_elem}' \n")
                f.write(f"        dsys,1 \n")
                f.write(f"        /view,1,0,-1,0 \n")
                f.write(f"        /vup,1,z \n")
                f.write(f"        /ang,1 \n")
                f.write(f"        /FOC,   1,   12.8385670721    ,   3.94929454584    ,  -10.2609282309 !0deg \n")
                f.write('         /DIST,1,4' + '\n')
                f.write('         /AUTO,1' + '\n')
                f.write('         /REP,FAST' + '\n')
            counter += 1
        f.write(f'    *endif \n')
    else:
        f.write("        /input,Input_macro_lineplan-11_" + element + ",inp,'" + path_Macros + "'" + '\n')
        f.write('        /REP,FAST' + '\n')
        f.write('        /VIEW,1,0,0,1' + '\n')
        f.write('        /VUP,,-x' + '\n')
        f.write('        /ANG,1' + '\n')
        f.write('        /ANG,1,90,ZS,1' + '\n')
        f.write('        /AUTO,1' + '\n')
        f.write('        /REP,FAST' + '\n')

    # Define the /PLOPTS options
    plopts_options = {
        'INFO': 3,
        'LEG1': 1,
        'LEG2': 1,
        'LEG3': 1,
        'FRAME': 1,
        'TITLE': 1,
        'MINM': 0,
        'FILE': 0,
        'LOGO': 0,
        'WINS': 1,
        'WP': 0,
        'DATE': 0
    }

    # Write the /PLOPTS options
    f.write('\n')
    f.write('        ! Image format definition:\n')
    for option, value in plopts_options.items():
        f.write(f"        /PLOPTS,{option},{value}\n")
    f.write(f"        /TRIAD,OFF\n")
    f.write(f"        /UDOC,1,cntr,bottom\n")

    # Define the /RGB options
    rgb_options = {
        'INDEX_1': '100,100,100,0',
        'INDEX_2': '0,0,0,15'
    }

    # Write the /RGB options
    for option, value in rgb_options.items():
        f.write(f"        /RGB,INDEX,{value}\n")

    # Write the remaining lines
    f.write('\n')
    f.write('        ! Image format definition:\n')
    f.write('\n')
    f.write('        /edge,1,1,\n')
    f.write('        /gformat,F,3,2\n')
    f.write('        /pnum,sval,0 ! 1 or 0 to show the values\n')
    f.write('        /number,0\n')
    f.write('        /graphics,power\n')
    f.write('        /efacet,1\n')
    f.write('        /gfile,2400\n')
    if re.search(r'LB', element):
        f.write('        cmsel,,LB\n')
        f.write('        nsle\n')
        f.write('        /rep\n')

    qnt_list = ['AXIUF', 'AXSUF', 'AYIUF', 'AYSUF', 'ATRUF', 'LANDA_N', 'LANDA_N', 'AXI', 'AXS', 'AYI', 'AYS', 'ATR']


    f.write('\n')
    f.write('        *DO,qNum,1,12,1' + '\n')
    for qNum in range(1,13):
        if qNum == 1:
            f.write(f'                *if,qNum,EQ,{qNum},then' + '\n')
        else:
            f.write(f'                *elseif,qNum,EQ,{qNum},then' + '\n')
        f.write(f"                        qnt = '{qnt_list[qNum - 1]}'" + "\n")
        f.write(f'                        pletab,{qnt_list[qNum - 1]},NOAVG' + '\n')
        f.write('                        *GET,MaxCont,PLNSOL,0,MAX' + '\n')
    f.write('                *endif' + '\n')
    f.write(f'\n')


    qNum_dict = {
        1: {
            'qNum': [1,2,3,4,5],
            'CVAL': '/CVAL,0.0,0.1,0.2,0.4,0.6,0.8,0.9,1,100,',
            'SMIN_COLOR': 'BLUE',
            'CNTR_COLOR': [('BLUE', 1), ('CBLU', 2), ('CYAN', 3), ('GREE', 4), ('YGRE', 5), ('YELL', 6), ('RED', 7),
                           ('LGRA', 8)],
            'SMAX_COLOR': 'LGRA',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        2: {
            'qNum': {6: True},
            'CVAL': '/CVAL,1,0.25,0.50,0.75,%target_margin%,5,10,15,100',
            'SMIN_COLOR': 'RED',
            'CNTR_COLOR': [('RED', 1), ('MRED', 2), ('ORAN', 3), ('LGRAY', 4), ('LGRAY', 5), ('LGRAY', 6), ('LGRAY', 7),
                           ('DGRA', 8)],
            'SMAX_COLOR': 'DGRAY',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        3: {
            'qNum': {6: False},
            'CVAL': '/CVAL,1,0.25,0.50,0.75,1,%target_margin%,5,10,100',
            'SMIN_COLOR': 'RED',
            'CNTR_COLOR': [('RED', 1), ('MRED', 2), ('ORAN', 3), ('YELL', 4), ('BLUE', 5), ('LGRAY', 6), ('LGRAY', 7),
                           ('DGRA', 8)],
            'SMAX_COLOR': 'DGRAY',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        4: {
            'qNum': {7: True},
            'CVAL': '/CVAL,0,%target_margin%,1.25,1.5,2,2.5,3,5,MaxCont',
            'SMIN_COLOR': 'LGRA',
            'CNTR_COLOR': [('DGRA', 1), ('YGRE', 2), ('GREE', 3), ('GCYA', 4), ('CYAN', 5), ('CBLU', 6), ('BLUE', 7),
                           ('BMAG', 8)],
            'SMAX_COLOR': 'BMAG',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        5: {
            'qNum': {7: False},
            'CVAL': '/CVAL,0,1,%target_margin%,1.25,1.5,2,2.5,4,MaxCont',
            'SMIN_COLOR': 'LGRA',
            'CNTR_COLOR': [('DGRA', 1), ('YGRE', 2), ('GREE', 3), ('GCYA', 4), ('CYAN', 5), ('CBLU', 6), ('BLUE', 7),
                           ('BMAG', 8)],
            'SMAX_COLOR': 'BMAG',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        6: {
            'qNum': [8,9,10,11],
            'CVAL': '/CVAL,1,15.7,40.2,62.83,94.3,125.6,165.9,250,300,',
            'SMIN_COLOR': 'BLAC',
            'CNTR_COLOR': [('BLAC', 1), ('LGRA', 2), ('DGRA', 3), ('BLUE', 4), ('GREE', 5), ('YELL', 6), ('ORAN', 7),
                           ('RED', 8)],
            'SMAX_COLOR': 'BMAG',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
        7: {
            'qNum': [12],
            'CVAL': '/CVAL,1,5.7,14.7,28.3,33.9,38.5,45.2,53.8,100,',
            'SMIN_COLOR': 'BLAC',
            'CNTR_COLOR': [('BLAC', 1), ('LGRA', 2), ('DGRA', 3), ('BLUE', 4), ('GREE', 5), ('YELL', 6), ('ORAN', 7),
                           ('RED', 8)],
            'SMAX_COLOR': 'BMAG',
            'RGB': [('/RGB,INDEX,100,100,100,0',), ('/RGB,INDEX,0,0,0,15',)]
        },
    }
    has_written_if_less_6 = False  # initialize flag variable
    has_written_if_greater_7 = False # initialize flag variable
    position_dict = 1  # initialize position counter
    for i in range(1, 13):
        if i < 6 and not has_written_if_less_6:
            f.write(f'                *if,qNum,lt,6,then' + '\n')
            f.write(f"                      {qNum_dict[position_dict]['CVAL']}" + '\n')
            f.write(f"                      /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
            for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                f.write(f"                      /COLOR,CNTR,{color},{value}" + "\n")
            f.write(f"                      /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
            for RGB in qNum_dict[position_dict]['RGB']:
                f.write(f"                      {RGB[0]}" + "\n")  # Remove quotes around RGB value
            has_written_if_less_6 = True  # set flag variable to True
            position_dict += 1  # increment the position counter
        elif 6 in qNum_dict[position_dict]['qNum']:
            if qNum_dict[position_dict]['qNum'][6] == True:
                f.write(f'                *elseif,qNum,eq,6,then' + '\n')
                f.write(f'                      *if,target_margin,eq,1,then' + '\n')
                f.write(f"                          {qNum_dict[position_dict]['CVAL']}" + '\n')
                f.write(f"                          /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
                for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                    f.write(f"                          /COLOR,CNTR,{color},{value}" + "\n")
                f.write(f"                          /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
                for RGB in qNum_dict[position_dict]['RGB']:
                    f.write(f"                          {RGB[0]}" + "\n")  # Remove quotes around RGB value
                position_dict += 1  # increment the position counter
            elif qNum_dict[position_dict]['qNum'][6] == False:
                f.write(f'                      *else' + '\n')
                f.write(f"                          {qNum_dict[position_dict]['CVAL']}" + '\n')
                f.write(f"                          /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
                for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                    f.write(f"                          /COLOR,CNTR,{color},{value}" + "\n")
                f.write(f"                          /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
                for RGB in qNum_dict[position_dict]['RGB']:
                    f.write(f"                          {RGB[0]}" + "\n")  # Remove quotes around RGB value
                f.write(f'                      *endif' + '\n')
                position_dict += 1  # increment the position counter
                # print(position_dict)
        elif 7 in qNum_dict[position_dict]['qNum']:
            if qNum_dict[position_dict]['qNum'][7] == True:
                f.write(f'                *elseif,qNum,eq,7,then' + '\n')
                f.write(f'                      *if,target_margin,eq,1,then' + '\n')
                f.write(f"                          *GET,MaxCont,PLNSOL,0,MAX" + '\n')
                f.write(f"                          {qNum_dict[position_dict]['CVAL']}" + '\n')
                f.write(f"                          /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
                for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                    f.write(f"                          /COLOR,CNTR,{color},{value}" + "\n")
                f.write(f"                          /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
                for RGB in qNum_dict[position_dict]['RGB']:
                    f.write(f"                          {RGB[0]}" + "\n")  # Remove quotes around RGB value
                position_dict += 1  # increment the position counter
            elif qNum_dict[position_dict]['qNum'][7] == False:
                f.write(f'                      *else' + '\n')
                f.write(f"                          *GET,MaxCont,PLNSOL,0,MAX" + '\n')
                f.write(f"                          {qNum_dict[position_dict]['CVAL']}" + '\n')
                f.write(f"                          /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
                for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                    f.write(f"                          /COLOR,CNTR,{color},{value}" + "\n")
                f.write(f"                          /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
                for RGB in qNum_dict[position_dict]['RGB']:
                    f.write(f"                          {RGB[0]}" + "\n")  # Remove quotes around RGB value
                f.write(f'                      *endif' + '\n')
                position_dict += 1  # increment the position counter
        elif i > 7 and not has_written_if_greater_7:
            f.write(f'                *elseif,qNum,gt,7,then' + '\n')
            f.write(f"                      {qNum_dict[position_dict]['CVAL']}" + '\n')
            f.write(f"                      /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
            for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                f.write(f"                      /COLOR,CNTR,{color},{value}" + "\n")
            f.write(f"                      /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
            for RGB in qNum_dict[position_dict]['RGB']:
                f.write(f"                      {RGB[0]}" + "\n")  # Remove quotes around RGB value
            has_written_if_greater_7 = True  # set flag variable to True
            position_dict += 1  # increment the position counter
        elif i == 12:
            f.write(f'                *elseif,qNum,eq,12,then' + '\n')
            f.write(f"                      {qNum_dict[position_dict]['CVAL']}" + '\n')
            f.write(f"                      /COLOR,SMIN,{qNum_dict[position_dict]['SMIN_COLOR']}" + '\n')
            for color, value in qNum_dict[position_dict]['CNTR_COLOR']:
                f.write(f"                      /COLOR,CNTR,{color},{value}" + "\n")
            f.write(f"                      /COLOR,SMAX,{qNum_dict[position_dict]['SMAX_COLOR']}" + '\n')
            for RGB in qNum_dict[position_dict]['RGB']:
                f.write(f"                      {RGB[0]}" + "\n")  # Remove quotes around RGB value
            has_written_if_greater_7 = True  # set flag variable to True
            position_dict += 1  # increment the position counter
    f.write(f'                *endif' + '\n')

# Plotting section

    #define a dictionary to store the qNum for each position
    qNum_position_dict = {
        1: {'qNum': ['AXIUF'], 'qnt': 'AXIUF', 'title': 'Bending'},
        2: {'qNum': ['AXSUF'], 'qnt': 'AXSUF', 'title': 'Bending'},
        3: {'qNum': ['AYIUF'], 'qnt': 'AYIUF', 'title': 'Bending'},
        4: {'qNum': ['AYSUF'], 'qnt': 'AYSUF', 'title': 'Bending'},
        5: {'qNum': ['ATRUF'], 'qnt': 'ATRUF', 'title': 'Shear'},
        6: {'qNum': ['LANDA_N'], 'qnt': 'LANDA_N', 'title': 'Landa_N'},
        7: {'qNum': ['LANDA_N'], 'qnt': 'LANDA_N', 'title': 'Landa_N'},
        8: {'qNum': ['AXI'], 'qnt': 'AXI', 'title': 'Bending'},
        9: {'qNum': ['AXS'], 'qnt': 'AXS', 'title': 'Bending'},
        10: {'qNum': ['AYI'], 'qnt': 'AYI', 'title': 'Bending'},
        11: {'qNum': ['AYS'], 'qnt': 'AYS', 'title': 'Bending'},
        12: {'qNum': ['ATR'], 'qnt': 'ATR', 'title': 'Shear'},
    }
    for qNum in qNum_position_dict:
        if qNum == 1:
            f.write(f'                *IF,qNum,EQ,{qNum},THEN\n')
        else:
            f.write(f'                *ELSEIF,qNum,EQ,{qNum},THEN\n')
        if 'BIO' in element or 'CROWN' in element:
            # f.write(f'')
            f.write(f'                        cmsel,,%reinf_comp% \n')
            f.write(f'                        nsle \n')
            f.write(f'                        /REP,FAST \n')
        elif 'CROWN' in element:
            f.write(f'')
        else:
            f.write(f'                        cmsel,,{element}\n')
        if element != 'B2' and element != 'LB':
            f.write('                        /show,png\n')
            f.write(
                f'                        /title, Safety factor in %name% -%reinf_comp%- {qNum_position_dict[qNum]["title"]}\n')
        else:
            f.write('                        *IF,conf,EQ,2,THEN\n')
            f.write(f'                           cmsel,,{element}_C2\n')
            f.write('                        *ENDIF\n')
            f.write('                        /show,png\n')
            f.write(
                f'                        /title, Safety factor in %name% -%reinf_comp%- {qNum_position_dict[qNum]["title"]}\n')

        for q in qNum_position_dict[qNum]['qNum']:
            f.write(f"                        qnt = '{q}'\n")
            f.write(f'                        pletab,{q},NOAVG\n')
            f.write('                        /show,close\n')
            if element != 'B2' and element != 'LB':
                if 'LANDA_N' in q:
                    if qNum == 6:
                        f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                        f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                        f.write(f"                        bulk_kb = BULK*1024\n")
                        f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                        if 'Bio' in element:
                            f.write(f'                                    /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                        else:
                            f.write(f'                                    /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                        f.write(
                            f"                                    /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                        f.write(f'                                    /out\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_positive,png\n')
                        f.write(f'                            *else\n')
                        f.write(f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_positive,png\n')
                        f.write(f'                            *endif\n')
                    else:
                        f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                        f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                        f.write(f"                        bulk_kb = BULK*1024\n")
                        f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                        if 'Bio' in element:
                            f.write(f'                                /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                        else:
                            f.write(
                                f'                                /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                        f.write(
                            f"                                /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                        f.write(f'                                /out\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_negative,png\n')
                        f.write(f'                            *else\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_negative,png\n')
                        f.write(f'                            *endif\n')
                else:
                    f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                    f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                    f.write(f"                        bulk_kb = BULK*1024\n")
                    f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                    if 'Bio' in element:
                        f.write(f'                                /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                    else:
                        f.write(
                            f'                                /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                    f.write(
                        f"                                /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                    f.write(f'                                /out\n')
                    f.write(
                        f'                                /rename,file000,png,,%name%_%reinf_comp%_{q},png\n')
                    f.write(f'                            *else\n')
                    f.write(
                        f'                                /rename,file000,png,,%name%_%reinf_comp%_{q},png\n')
                    f.write(f'                            *endif\n')
            else:
                if 'LANDA_N' in q:
                    if qNum == 7:
                        f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                        f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                        f.write(f"                        bulk_kb = BULK*1024\n")
                        f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                        if 'Bio' in element:
                            f.write(f'                                /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                        else:
                            f.write(
                                f'                                /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                        f.write(
                            f"                                /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                        f.write(f'                                /out\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_positive,png\n')
                        f.write(f'                            *else\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_positive,png\n')
                        f.write(f'                            *endif\n')
                    else:
                        f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                        f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                        f.write(f"                        bulk_kb = BULK*1024\n")
                        f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                        if 'Bio' in element:
                            f.write(f'                                /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                        else:
                            f.write(
                                f'                                /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                        f.write(
                            f"                                /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                        f.write(f'                                /out\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_negative,png\n')
                        f.write(f'                            *else\n')
                        f.write(
                            f'                                /rename,file000,png,,%name%_%reinf_comp%_{q}_negative,png\n')
                        f.write(f'                            *endif\n')
                else:
                    f.write(f'                        /INQUIRE,AVAILABLE,EXIST,file000,png,\n')
                    f.write(f'                        /INQUIRE,BULK,SIZE,file000,png,\n')
                    f.write(f"                        bulk_kb = BULK*1024\n")
                    f.write(f"                            *if, bulk_kb, lt, 10, or, AVAILABLE, eq, 0, then\n")
                    if 'Bio' in element:
                        f.write(f'                                /out,.\{input_folder}\BIO\warnings,txt,,append\n')
                    else:
                        f.write(
                            f'                                /out,.\{input_folder}\{element}\warnings,txt,,append\n')
                    f.write(
                        f"                                /com, %name%\tPICTURE missing or wrong:{q}:\t%reinf_comp%\n")
                    f.write(f'                                /out\n')
                    f.write(
                        f'                                /rename,file000,png,,%name%_%reinf_comp%_{q},png\n')
                    f.write(f'                            *else\n')
                    f.write(
                        f'                                /rename,file000,png,,%name%_%reinf_comp%_{q},png\n')
                    f.write(f'                            *endif\n')
    f.write('                *ENDIF' + '\n')
    f.write('        *ENDDO' + '\n')
    f.write('   *enddo' + '\n')
    f.write('*enddo' + '\n')

    # Create the directory for the element if it doesn't exist
    element_dir = os.path.join(path_Macros, input_folder, element)
    if not os.path.exists(element_dir):
        os.makedirs(element_dir)

    # Create the CW and RW directories for crown elements if they don't exist
    if 'CROWN' in element:
        for item in ['CW', 'RW']:
            item_dir = os.path.join(element_dir, item)
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)

    if 'Bio' in element:
        f.write(f'/sys,move,{path_Macros}*.png \t {path_Macros}' + f'{input_folder}' + '\BIO\n')
    elif 'CROWN' in element:
        f.write(f'/sys,move,{path_Macros}*RW*.png \t {path_Macros}' + f'{input_folder}' + '\CROWN\RW\n')
        f.write(f'/sys,move,{path_Macros}*CW*.png \t {path_Macros}' + f'{input_folder}' + '\CROWN\CW\n')
    else:
        f.write(f'/sys,move,{path_Macros}*.png \t {path_Macros}' + f'{input_folder}' + '\\' + f'{element}\n')
    f.close()

    # write_grid_lines(lineplan, element)

sys.exit()
