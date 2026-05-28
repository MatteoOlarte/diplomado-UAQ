# pyright: reportAttributeAccessIssue=false

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing()
data = pd.DataFrame(cali.data, columns=cali.feature_names)

# ----------------------------------------------------------------------------------------------------------------------
#  Recesion Lineal: Variable objetivo: MedHouseVal
# ----------------------------------------------------------------------------------------------------------------------

data['MedHouseVal'] = cali.target
data.head(5)

data.info()

data['MedHouseVal'].min()
data['MedHouseVal'].max()

# ----------------------------------------------------------------------------------------------------------------------
# 1. Visualización de la variable objetivo
# ----------------------------------------------------------------------------------------------------------------------

# Se va a mostrar el histograma de cada una de las variables del dataset, se marco con un color azul la variable
# objetivo para que sea facilmente identificable.

cols = 4
n_plots = len(data.columns)
rows = math.ceil(n_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
axes = axes.ravel()

for i, column in enumerate(data.columns):
    plot_color = 'steelblue' if column == 'MedHouseVal' else 'lightslategray'

    data[column].hist(ax=axes[i], bins=30, color=plot_color, edgecolor='black')
    axes[i].set_title(f'Histograma - {column}', fontsize=14)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frecuencia')

for i in range(n_plots, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 2. Matriz de Correlación
# ----------------------------------------------------------------------------------------------------------------------

# * La variable que mas ajecta a "MedHouseVal" es "MedInc" con una correlacion de 0.69; siendo una correlacion positiva.

# * Variables como "HouseAge", "AveRooms" tienen una correlacion positiva pero menor a "MedInc".

# * Longitud y Latitud son coordenadas geográficas, se puede decir que si la casa esa ubicada en la costa o en ciudades
#   importantes (SF o LA) el valor de la casa es mayor.

plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Matriz de Correlación')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 3. Visualización de la relación entre datos
# ----------------------------------------------------------------------------------------------------------------------

# Ubicacion Geografica vs MedHouseVal
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=data['Longitude'],
    y=data['Latitude'],
    hue=data['MedHouseVal'],
    palette='RdYlGn',
    alpha=0.7
)
plt.title('Ubicación Geográfica vs MedHouseVal')
plt.show()

# HouseAge vs MedHouseVal
x = data["MedInc"]
y = data["MedHouseVal"]

m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y, alpha=0.15, s=8, color="#185FA5", label="datos")
ax.plot(x_line, m * x_line + b, color="#185FA5", linewidth=2, label="tendencia")
ax.set_xlabel("MedInc", fontsize=11)
ax.set_ylabel("MedHouseVal", fontsize=11)
ax.set_title("MedInc vs MedHouseVal  |  r = 0.69 — correlación fuerte", fontsize=11)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("scatter_medinc.png", dpi=150, bbox_inches="tight")
plt.show()



# =============================================================================

# Visualiza la relación entre una característica de tu elección (por ejemplo, HouseAge) y la variable objetivo (MedHouseVal).
# Elimina datos no útiles (por ejemplo, valores nulos). Na
# Elimina datos atípicos utilizando un umbral de 3 desviaciones estándar.
# Entrena un modelo de regresión lineal utilizando el conjunto de entrenamiento seleccionando una característica y el regresos de sklearn.
# Calcula el error cuadrático medio (MSE) y el coeficiente de determinación (R²) para evaluar el rendimiento del modelo.
# Visualiza los resultados de la predicción en comparación con los datos reales.
# Muestra el intercepto y los coeficientes de la regresión lineal.
# Presenta una tabla con los coeficientes de cada característica.
# Entrena y evalúa otro modelo de regresión lineal utilizando diferentes características (por ejemplo, MedInc, HouseAge, AveRooms).
# Compara los resultados con el modelo anterior.


# ¿Cómo se comparan los errores cuadráticos medios (MSE) de los dos modelos?
# ¿Cuál de los dos modelos tiene un mejor rendimiento y por qué?
# ¿Qué puedes concluir sobre la importancia de seleccionar las características adecuadas para el modelo de regresión lineal?

# =============================================================================
