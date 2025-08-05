import psycopg2
import pandas as pd

#1-Genera una conexión a la base de datos Chinook y ejecuta una consulta. Recibe como parámetro una consulta SQL y devuelve un DataFrame con los resultados
def conexion_chinook(sql_query):

    conn = psycopg2.connect (
        host="localhost",
        database="Chinook",
        user = "postgres",
        password = "tornado88")
#2-Generar un cursor para ejecutar la consulta, query que le pase, ejecuta registros y devuelve variable rows 
    cursor = conn.cursor()

    cursor.execute(sql_query)
    rows = cursor.fetchall()
#3-Generar un DataFrame llamado tabla con los resultados de la consulta y renombro las columnas del DF.
    tabla = pd.DataFrame(rows)
    colnames = [desc[0] for desc in cursor.description]

    tabla = tabla.set_axis(colnames, axis = 1)

    cursor.close()
    conn.close() 

    return tabla