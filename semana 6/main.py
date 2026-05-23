import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# convertir grados farenheit a celsious

# definimos las entradas y las salidas
F = np.array([-25, 0, 50, 75, 130, 200, 220, 300])  # entradas
C = np.array([-31.66, -17.77, 10, 23.88, 54.44, 93.33, 104.44, 148.88])  # salida

# creamos un perceptron (solo capa entrada y capa de salida)
layer = tf.keras.layers.Dense(units=1, input_shape=[1])  # dim salida y dim entrada
modelo = tf.keras.Sequential([layer])

# compilar el modelo
modelo.compile(loss='mean_squared_error')

# entrenamos
historial = modelo.fit(F, C, epochs=3000, verbose=True)

# visualicemos la perdida
plt.plot(historial.history['loss'])
plt.show()

# testing
valor = np.array([175])  # valor a predecir
test = modelo.predict(valor)
print(f"El valor de {valor} grados farenheit es igual a {test} grados celsious")


# ----------------------------------------------------------------------------------------------------------------------
# Un perseptron simple, con con 100 epocas el error es muy grande.
# Un perseptron simple, con con 1K epocas el error es cercano a 0.
# Un perseptron simple, con con 3K epocas el error es cercano a 0.
# ----------------------------------------------------------------------------------------------------------------------
