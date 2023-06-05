import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo de Excel
archivo_excel = r'\\iter\cfs\partners\ENGAGE\_INT\02.DESIGN\15.CD\62.PBS 62\11.TKB\04.CW\105. TSS 2022\06.3 ENG_51_CL_110076_CW_B2\v08\reinforcement\B2_Reinforcement_v77_mod31.05.2023.xlsm'

# Ruta del archivo CSV
archivo_csv = r'D:\Old_C_Drive\ANSYS_WORK\55_TSS_update\01_Methodology\03_ANSYS\temp\ANSYS_repo\trunk\20_TSS_2022\lever_arm\2023-05-31\graphs_inner_wo_ASB_base.csv'

# Diccionario con el número de fila por hoja de Excel y columna del archivo CSV
filas_interes = {
    'RAXI (CX)': (30, 'AREA_XI'),
    'RAXI (CX) EXTRA': (31, 'AREA_XI'),
    'RAXS (CX)': (35, 'AREA_XS'),
    'RAYI (CX) ': (20, 'AREA_YI'),
    'RAYI (CX) EXTRA': (21, 'AREA_YI'),
    'RAYS (CX)': (15, 'AREA_YS'),
    'RAYS (CX) EXTRA': (15, 'AREA_YS')
}

# Leer el archivo CSV y saltar las 2 primeras líneas
df_csv = pd.read_csv(archivo_csv, skiprows=1, usecols=[0, 1, 2, 3, 4, 5], names=['Elements', 'AREA_XI', 'AREA_XS', 'AREA_YI', 'AREA_YS', 'Distance'], skipinitialspace=True, encoding='ISO-8859-1')
df_csv['AREA_XI'] = pd.to_numeric(df_csv['AREA_XI'], errors='coerce').round(2)
df_csv['AREA_XS'] = pd.to_numeric(df_csv['AREA_XS'], errors='coerce').round(2)
df_csv['AREA_YI'] = pd.to_numeric(df_csv['AREA_YI'], errors='coerce').round(2)
df_csv['AREA_YS'] = pd.to_numeric(df_csv['AREA_YS'], errors='coerce').round(2)
df_csv['Distance'] = pd.to_numeric(df_csv['Distance'], errors='coerce')

# Reindexar el DataFrame df_csv
df_csv = df_csv.dropna().reset_index(drop=True)
df_csv = df_csv.sort_values('Distance').reset_index(drop=True)
df_csv['Elemento'] = df_csv.index


# Obtener los valores ordenados de la columna Elements como distancias interpoladas
distancias_interpoladas = df_csv['Distance'].values

# Crear una lista para almacenar los DataFrames intermedios
lista_df_interpolados = []

# Crear el diccionario para almacenar los valores interpolados
valores_interpolados = {'Distancia': distancias_interpoladas}

# Iterar sobre las hojas de Excel y el número de fila y columna correspondiente
for hoja_excel, (fila_interes, columna_csv) in filas_interes.items():
    # Leer los datos del archivo de Excel
    df = pd.read_excel(archivo_excel, sheet_name=hoja_excel, header=None)

    # Filtrar las filas de interés (fila 2 y fila correspondiente)
    fila_distancia = df.iloc[1, 11:]  # Cambiar el índice de columna según sea necesario
    fila_armado = df.iloc[fila_interes, 11:]  # Cambiar el índice de columna según sea necesario

    # Convertir los valores de distancia y armado a listas
    distancias_datos = list(fila_distancia.astype(float))
    armados_datos = list(fila_armado.astype(float))

    # Realizar la interpolación lineal para las 100 distancias
    valores_interpolados[hoja_excel] = np.interp(distancias_interpoladas, distancias_datos, armados_datos)

    # Crear la gráfica para la curva original y los valores interpolados
    plt.plot(distancias_datos, armados_datos, label='Curva original')
    plt.plot(distancias_interpoladas, valores_interpolados[hoja_excel], '--', label='Interpolación')

    # Obtener los valores de la columna correspondiente del archivo CSV
    columna_csv_valores = df_csv[columna_csv].values

    # Agregar los valores de la columna CSV a la gráfica
    plt.scatter(distancias_interpoladas, columna_csv_valores, color='red', label='Columna CSV')

    # Agregar etiquetas con los valores de armado para cada 2000 mm aproximadamente
    etiquetas_x = np.linspace(0, 30000, 8)  # Generar valores x para las etiquetas
    for x, y in zip(etiquetas_x, valores_interpolados[hoja_excel][::12]):
        label = f'{y:.2f}'  # Formato de 2 decimales
        plt.text(x, y, label, ha='center', va='bottom')

    plt.xlabel('Distancia (mm)')
    plt.ylabel('Armado (cm²/m)')
    plt.title(f'Gráfica de Armado vs Distancia ({hoja_excel})')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'\\iter\cfs\partners\ENGAGE\_INT\02.DESIGN\15.CD\62.PBS 62\11.TKB\04.CW\105. TSS 2022\06.3 ENG_51_CL_110076_CW_B2\v08\reinforcement\ANSYS\{}.png'.format(hoja_excel),dpi=600)
    # plt.show()
    plt.close()

    # Comparar los valores interpolados con los del archivo CSV y marcar las diferencias
    df_interpolados = pd.DataFrame(valores_interpolados)

    for i, fila in df_interpolados.iterrows():
        area_xi_csv = df_csv.at[i, 'AREA_XI']
        area_xs_csv = df_csv.at[i, 'AREA_XS']
        area_yi_csv = df_csv.at[i, 'AREA_YI']
        area_ys_csv = df_csv.at[i, 'AREA_YS']

        df_interpolados.at[i, 'AREA_XI_CSV'] = area_xi_csv
        df_interpolados.at[i, 'AREA_XS_CSV'] = area_xs_csv
        df_interpolados.at[i, 'AREA_YI_CSV'] = area_yi_csv
        df_interpolados.at[i, 'AREA_YS_CSV'] = area_ys_csv

    # Agregar el número de elemento al DataFrame df_interpolados
    df_interpolados.insert(0, 'Elemento', df_csv['Elements'])

    # Guardar los datos interpolados y la indicación de diferencia en el archivo TSV
    ruta_tsv = r'\\iter\cfs\partners\ENGAGE\_INT\02.DESIGN\15.CD\62.PBS 62\11.TKB\04.CW\105. TSS 2022\06.3 ENG_51_CL_110076_CW_B2\v08\reinforcement\ANSYS\interpolated_data.tsv'
    df_interpolados.round(2).to_csv(ruta_tsv, sep='\t', index=False)


# Leer el archivo TSV
df_interpolados = pd.read_csv(ruta_tsv, sep='\t')

# Calcular las diferencias y agregar las columnas correspondientes
df_interpolados['Diferencia_RAXI'] = df_interpolados['RAXI (CX)'] - df_interpolados['AREA_XI_CSV']
df_interpolados['Diferencia_RAXI_EXTRA'] = df_interpolados['RAXI (CX) EXTRA'] - df_interpolados['AREA_XI_CSV']
df_interpolados['Diferencia_RAXS'] = df_interpolados['RAXS (CX)'] - df_interpolados['AREA_XS_CSV']
df_interpolados['Diferencia_RAYI'] = df_interpolados['RAYI (CX) '] - df_interpolados['AREA_YI_CSV']
df_interpolados['Diferencia_RAYI_EXTRA'] = df_interpolados['RAYI (CX) EXTRA'] - df_interpolados['AREA_YI_CSV']
df_interpolados['Diferencia_RAYS'] = df_interpolados['RAYS (CX)'] - df_interpolados['AREA_YS_CSV']
df_interpolados['Diferencia_RAYS_EXTRA'] = df_interpolados['RAYS (CX) EXTRA'] - df_interpolados['AREA_YS_CSV']

# Aplicar condición para asignar valor 0 o la diferencia
df_interpolados['Diferencia_RAXI'] = df_interpolados['Diferencia_RAXI'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAXI_EXTRA'] = df_interpolados['Diferencia_RAXI_EXTRA'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAXS'] = df_interpolados['Diferencia_RAXS'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAYI'] = df_interpolados['Diferencia_RAYI'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAYI_EXTRA'] = df_interpolados['Diferencia_RAYI_EXTRA'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAYS'] = df_interpolados['Diferencia_RAYS'].apply(lambda x: 0 if abs(x) < 2 else x)
df_interpolados['Diferencia_RAYS_EXTRA'] = df_interpolados['Diferencia_RAYS_EXTRA'].apply(lambda x: 0 if abs(x) < 2 else x)


# Guardar el DataFrame actualizado en el archivo TSV
df_interpolados.round(2).to_csv(ruta_tsv, sep='\t', index=False)

import pandas as pd

# Leer el archivo TSV original
ruta_tsv = r'\\iter\cfs\partners\ENGAGE\_INT\02.DESIGN\15.CD\62.PBS 62\11.TKB\04.CW\105. TSS 2022\06.3 ENG_51_CL_110076_CW_B2\v08\reinforcement\ANSYS\interpolated_data.tsv'
df_original = pd.read_csv(ruta_tsv, sep='\t')

# Ruta salida
ruta_corregido_tsv = r'\\iter\cfs\partners\ENGAGE\_INT\02.DESIGN\15.CD\62.PBS 62\11.TKB\04.CW\105. TSS 2022\06.3 ENG_51_CL_110076_CW_B2\v08\reinforcement\ANSYS\interpolated_corrected_data.tsv'

# Crear una copia del DataFrame original
df_modificado = df_original.copy()

# Iterar fila a fila y aplicar las modificaciones
for i in range(len(df_modificado)):
    if (df_modificado.at[i, 'Diferencia_RAXI'] == df_modificado.at[i, 'Diferencia_RAXI_EXTRA']):
        df_modificado.at[i, 'Diferencia_RAXI'] = df_modificado.at[i, 'RAXI (CX)']
        df_modificado.at[i, 'Diferencia_RAXI_EXTRA'] = df_modificado.at[i, 'RAXI (CX) EXTRA']
    df_modificado.at[i, 'Diferencia_RAXS'] = df_modificado.at[i, 'RAXS (CX)']
    if (df_modificado.at[i, 'Diferencia_RAYI'] == df_modificado.at[i, 'Diferencia_RAYI_EXTRA']):
        df_modificado.at[i, 'Diferencia_RAYI'] = df_modificado.at[i, 'RAYI (CX) ']
        df_modificado.at[i, 'Diferencia_RAYI_EXTRA'] = df_modificado.at[i, 'RAYI (CX) EXTRA']
    if (df_modificado.at[i, 'Diferencia_RAYS'] == df_modificado.at[i, 'Diferencia_RAYS_EXTRA']):
        df_modificado.at[i, 'Diferencia_RAYS'] = df_modificado.at[i, 'RAYS (CX)']
        df_modificado.at[i, 'Diferencia_RAYS_EXTRA'] = df_modificado.at[i, 'RAYS (CX) EXTRA']

# Crear el nuevo DataFrame con las columnas requeridas
nuevo_df = pd.DataFrame()
nuevo_df['NumeroElemento'] = df_original['Elemento']
nuevo_df['DistanciaAnterior'] = df_original['Distancia']
nuevo_df['RAXI (CX)'] = df_modificado['Diferencia_RAXI']
nuevo_df['RAXI (CX) EXTRA'] = df_modificado['Diferencia_RAXI_EXTRA']
nuevo_df['RAXS (CX)'] = df_modificado['Diferencia_RAXS']
nuevo_df['RAYI (CX)'] = df_modificado['Diferencia_RAYI']
nuevo_df['RAYI (CX) EXTRA'] = df_modificado['Diferencia_RAYI_EXTRA']
nuevo_df['RAYS (CX)'] = df_modificado['Diferencia_RAYS']
nuevo_df['RAYS (CX) EXTRA'] = df_modificado['Diferencia_RAYS_EXTRA']

# Guardar el nuevo DataFrame en el archivo "interpolated_corrected_data.tsv"
nuevo_df.round(2).to_csv(ruta_corregido_tsv, sep='\t', index=False)


