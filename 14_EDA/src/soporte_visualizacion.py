# Visualización

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualizacion_categoricas(df, categorical_columns):
    n_cols = 2  # Número de columnas en los subplots
    n_rows = -(-len(categorical_columns) // n_cols)  # Calcular el número de filas necesarias

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5), constrained_layout=True)

    # Aplanar el arreglo de ejes para iterar fácilmente
    axes = axes.flatten()

    for i, col in enumerate(categorical_columns):
        sns.countplot(data=df, x=col, ax=axes[i], palette="pastel", hue=col)
        axes[i].set_title(f"Bar Plot de {col}")
        axes[i].set_xlabel("Categorías")
        axes[i].set_ylabel("Frecuencia")
        axes[i].tick_params(axis='x', rotation=90)  # Rotar los nombres del eje X

    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()