## Funciones básicas a hacer, se ejecuta al comienzo para hacer preanálisis básico y rapido.

import pandas as pd
import numpy as np

def preanalisis(ruta_archivo):
    "docstring"
    df = carga_datos(ruta_archivo)
    set_opciones(df)
    resumen_datos(df)
    return df

def carga_datos(ruta_archivo):
    """
    Carga los datos desde un archivo CSV, Parquet o Excel.
    """
    # Carga de archivos
    try:
        if ruta_archivo.endswith(".csv"):
            df = pd.read_csv(ruta_archivo)
        elif ruta_archivo.endswith(".parquet"):
            df = pd.read_parquet(ruta_archivo)
        elif ruta_archivo.endswith(".xlsx") or ruta_archivo.endswith(".xls"):
            df = pd.read_excel(ruta_archivo)
        else:
            raise ValueError("Formato de archivo no soportado.")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None
    return df

def set_opciones(df):
    pass

def set_opciones(df):
    # Opciones de datos y columnas
    pd.set_option('display.max_columns', None)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df = df.map(lambda x: x.strip().replace(' ', '_').lower() if isinstance(x, str) else x)
    return df

def resumen_datos(df):
    # Resumen de los datos
    print("\n--- Resumen de datos ---")
    print(f"El número de filas es: {df.shape[0]}")
    print(f"El número de columnas es: {df.shape[1]}\n---------")
    print("Información general:")
    display(df.info()) # type: ignore
    print("\n")
    num_dupl = df.duplicated().sum()
    print(f"El número de duplicados: {num_dupl}")
    #se puede elimimar o comentar estas 3 línea si no se quiere eliminar duplicados
    if num_dupl != 0:
        print("¡Eliminamos los duplicados...")
        df.drop_duplicates(inplace=True)
        

    df_nulos = pd.DataFrame({"count": df.isnull().sum(), 
                            "% nulos": (df.isnull().sum() / df.shape[0]).round(3) * 100})

    df_nulos = df_nulos[df_nulos["count"] > 0]
    df_nulos_sorted = df_nulos.sort_values(by="% nulos", ascending=False)
    print("Los nulos que tenemos en el conjunto de datos son:")
    display(df_nulos_sorted) # type: ignore

    columnas_por_tipo = {dtype: df.select_dtypes(dtype).columns.tolist() for dtype in df.dtypes.unique()}
    print("Los tipos de las columnas son:")
    display(pd.DataFrame(df.dtypes, columns=["tipo_dato"])) # type: ignore
    print("\nColumnas agrupadas por tipo de dato:")
    for tipo, columnas in columnas_por_tipo.items():
        print(f"- {tipo}:", columnas)

    print("\n-------------------------\n")
    identificar_columnas_booleanas(df)
    print("-----------------")
    display(df.head()) # type: ignore

    return df

def identificar_columnas_booleanas(df):
    # Identificar columnas con valores únicos {0, 1}
    posibles_booleans = [col for col in df.columns if set(df[col].dropna().unique()) <= {0, 1}]
    print(f"Columnas que podrían ser booleanas: {posibles_booleans}")
    return posibles_booleans

