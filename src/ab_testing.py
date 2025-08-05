
# importamos las librerías que necesitamos

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualización
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluar linealidad de las relaciones entre las variables
# y la distribución de las variables
# ------------------------------------------------------------------------------
import scipy.stats as stats
from scipy.stats import norm, f_oneway

# Configuración
# -----------------------------------------------------------------------
pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames

# Gestión de los warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")


class AB_testing:

    """
    Clase para realizar pruebas de hipótesis A/B testing en un conjunto de datos. Evalúa la normalidad,
    homogeneidad de varianzas y selecciona el test estadístico adecuado (Mann-Whitney, Z o ANOVA).
    
    Atributos:
    ----------
    dataframe : pd.DataFrame
        Conjunto de datos con los valores a evaluar.
    col_grupos : str
        Nombre de la columna que identifica los grupos de control y prueba.
    lista_col_estudiar : list
        Lista de columnas métricas a evaluar para pruebas A/B.
    dicc_col : dict
        Diccionario que almacena los resultados de normalidad y homogeneidad para cada métrica.

    Métodos:
    --------
    __init__(dataframe, col_grupos, lista_col_estudiar)
        Inicializa la clase y evalúa normalidad y homogeneidad para cada columna en lista_col_estudiar.
    normalidad()
        Realiza la prueba de normalidad Shapiro-Wilk para cada métrica y almacena los resultados en dicc_col.
    homogeneidad()
        Realiza la prueba de homogeneidad de varianzas Levene para cada métrica y almacena los resultados en dicc_col.
    elegir_tests()
        Selecciona y ejecuta el test estadístico adecuado (Mann-Whitney, Z o ANOVA) para cada métrica basada en los resultados de normalidad y homogeneidad.
    test_man_whitney(columna_metrica)
        Ejecuta el test no paramétrico de Mann-Whitney para comparar medianas entre grupos en caso de datos no normales o varianzas no homogéneas.
    test_z(columna_metrica)
        Ejecuta la prueba Z para comparar medias entre dos grupos cuando los datos son normales y las varianzas son homogéneas.
    anova(columna_metrica)
        Ejecuta la prueba ANOVA para comparar medias entre tres o más grupos cuando los datos son normales y las varianzas son homogéneas.
    """

    def __init__ (self, dataframe, col_grupos, lista_col_estudiar):

        """
        Inicializa la clase AB_testing con un conjunto de datos y evalúa normalidad y homogeneidad.

        Parámetros:
        -----------
        dataframe : pd.DataFrame
            DataFrame que contiene los datos a evaluar.
        col_grupos : str
            Nombre de la columna que identifica los grupos de control y prueba.
        lista_col_estudiar : list
            Lista de columnas métricas a evaluar para pruebas A/B.
        """    

        self.dataframe = dataframe
        self.col_grupos = col_grupos
        self.lista_col_estudiar = lista_col_estudiar
        self.dicc_col = {}
        for col in lista_col_estudiar:
            self.dicc_col[col] = {}

        self.normalidad()
        self.homogeneidad()

        self.elegir_tests()
    



    def normalidad(self):

        """
        Evalúa la normalidad de cada métrica en lista_col_estudiar usando la prueba Shapiro-Wilk.
        
        Retorna:
        --------
        dict
            Diccionario con el resultado de normalidad de cada métrica (clave: métrica, valor: 's' o 'n').
        """
      
        for col in self.lista_col_estudiar:
            statistic, p_value = stats.shapiro(self.dataframe[col])
            if p_value > 0.05:
                print(f"Para la columna {col.upper()} los datos siguen una distribución normal.")
                self.dicc_col[col]['Normalidad'] = 's' 
            else:
                print(f"Para la columna {col.upper()} los datos no siguen una distribución normal.")
                self.dicc_col[col]['Normalidad'] = 'n' 

        print('\n-----------------------------\n')

        return self.dicc_col



    def homogeneidad (self):

        """
        Evalúa la homogeneidad de varianzas entre grupos para cada métrica usando la prueba de Levene.

        Retorna:
        --------
        dict
            Diccionario con el resultado de homogeneidad de varianzas de cada métrica (clave: métrica, valor: 's' o 'n').
        """
        

        for col in self.lista_col_estudiar:

            valores_evaluar = []
        
            for valor in self.dataframe[self.col_grupos].unique():
                valores_evaluar.append(self.dataframe[self.dataframe[self.col_grupos]== valor][col])

            statistic, p_value = stats.levene(*valores_evaluar)
            if p_value > 0.05:
                print(f"Para la métrica {col.upper()} las varianzas son homogéneas entre grupos.")
                self.dicc_col[col]['Homocedasticidad'] = 's'
            else:
                print(f"Para la métrica {col.upper()}, las varianzas no son homogéneas entre grupos.")
                self.dicc_col[col]['Homocedasticidad'] = 'n'

        print('\n-----------------------------\n')
        return self.dicc_col


    def elegir_tests(self):

        """
        Selecciona y ejecuta el test adecuado (Mann-Whitney, Z o ANOVA) para cada métrica basada en los resultados de normalidad y homogeneidad.
        """

        for col in self.lista_col_estudiar:

            if self.dicc_col[col]['Normalidad'] == 'n' or self.dicc_col[col]['Homocedasticidad'] == 'n':
                print(f'Para la columna {col.upper()} se va a realizar el test de MANN WHITNEY')

                self.test_man_whitney(col)
            
            elif len(self.dataframe[self.col_grupos].unique()) == 2:
                print(f'Para la columna {col.upper()} se va a realizar el test de Z-SCORE')

                self.test_z(col)
            elif len(self.dataframe[self.col_grupos].unique()) > 2:

                print(f'Para la columna {col.upper()} se va a realizar el test de ANOVA')
                self.anova(col)



    def test_man_whitney(self, columna_metrica):

        """
        Ejecuta el test de Mann-Whitney para comparar medianas entre dos grupos en caso de datos no normales o varianzas no homogéneas.

        Parámetros:
        -----------
        columna_metrica : str
            Nombre de la columna métrica a evaluar.
        """


        valores_grupos = self.dataframe[self.col_grupos].unique().tolist()

        control = self.dataframe[self.dataframe[self.col_grupos]== valores_grupos[0]]

        test = self.dataframe[self.dataframe[self.col_grupos]== valores_grupos[1]]
        
        
        metrica_control = control[columna_metrica]
        metrica_test = test[columna_metrica]

        u_statistic, p_value = stats.mannwhitneyu(metrica_control, metrica_test)
        
        if p_value < 0.05:
            print(f"Para la métrica {columna_metrica}, las medianas son diferentes.")
        else:
            print(f"Para la métrica {columna_metrica}, las medianas son iguales.")

        print('\n-----------------------------\n')
            

    def test_z (self,columna_metrica):

        """
        Ejecuta la prueba Z para comparar medias entre dos grupos cuando los datos son normales y las varianzas son homogéneas.

        Parámetros:
        -----------
        columna_metrica : str
            Nombre de la columna métrica a evaluar.
        """

        valores_col = self.dataframe[self.col_grupos].unique().tolist()

        media_1 = self.dataframe[self.dataframe[self.col_grupos] == valores_col[0]][columna_metrica].mean()
        std_1 = self.dataframe[self.dataframe[self.col_grupos] == valores_col[0]][columna_metrica].std()


        media_2 = self.dataframe[self.dataframe[self.col_grupos] == valores_col[1]][columna_metrica].mean()
        std_2 = self.dataframe[self.dataframe[self.col_grupos] == valores_col[1]][columna_metrica].std()


        n_1= len(self.dataframe[self.dataframe[self.col_grupos] == valores_col[0]])
        n_2 = len(self.dataframe[self.dataframe[self.col_grupos] == valores_col[1]])


        z_stat = (media_2 - media_1) / np.sqrt((std_1**2 / n_1) + (std_2**2 / n_2))


        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))


        alpha = 0.05
        if p_value < alpha:
            print("Hay una diferencia significativa entre el grupo de control y el grupo de prueba.")

        else:
            print("No hay evidencia de una diferencia significativa entre los grupos.")

        print('\n-----------------------------\n')


    def anova(self, columna_metrica):


        """
        Ejecuta la prueba ANOVA para comparar medias entre tres o más grupos cuando los datos son normales y las varianzas son homogéneas.

        Parámetros:
        -----------
        columna_metrica : str
            Nombre de la columna métrica a evaluar.
        """

        metricas_anova = []

        for grupo in self.dataframe[self.col_grupos].unique():

            metricas_anova.append(self.dataframe[self.dataframe[self.col_grupos] == grupo][columna_metrica])

        anova_resultado = f_oneway(*metricas_anova)


        alpha = 0.05
        if anova_resultado.pvalue < alpha:
            print("Hay diferencias significativas entre al menos dos grupos.")

        else:
            print("No hay evidencia de diferencias significativas entre los grupos.")

        print('\n-----------------------------\n')

        
    