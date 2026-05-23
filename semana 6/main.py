import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# convertir grados farenheit a celsious

# definimos las entradas y las salidas
F = np.array([-25, 0, 50, 75, 130, 200, 220, 300])  # entradas
C = np.array([-31.67, -17.78, 10, 23.89, 54.44, 93.33, 104.44, 148.89])  # salida

# creamos un perceptron (solo capa entrada y capa de salida)
input = tf.keras.layers.Input(shape=[1])
HL1 = tf.keras.layers.Dense(units=3)
HL2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([input, HL1, HL2, output])

# compilar el modelo
model.compile(loss='mean_squared_error')

# entrenamos
historial = model.fit(F, C, epochs=3000, verbose=True)

# visualicemos la perdida
plt.plot(historial.history['loss'])
plt.show()

# testing
valor = np.array([175])  # valor a predecir
test = model.predict(valor)
print(f"El valor de {valor} grados farenheit es igual a {test} grados celsious")


# ----------------------------------------------------------------------------------------------------------------------
# Un perseptron simple, con con 100 epocas el error es muy grande.
# Un perseptron simple, con con 1K epocas el error es cercano a 0.
# Un perseptron simple, con con 3K epocas el error es cercano a 0.
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Construimos un perseptron multicapa 2 capas ocultas y 3 neuronas por capa.
# ----------------------------------------------------------------------------------------------------------------------

