import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=2000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.03,
    random_state=42
)


plt.figure(figsize=(16, 8))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        # Inicializar pesos
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            # Propagación
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)

            # Cálculo de costo
            cost = self._compute_cost(h, y)
            self.cost_history.append(cost)

            # Actualización de pesos (descenso de gradiente)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


modelo = LogisticRegression(learning_rate=0.01, n_iterations=5000)
modelo.fit(X_train, y_train)


# visualicemos la funcion de costo

plt.plot(modelo.cost_history)
plt.grid()
plt.show()


# clasifiquemos

y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)


# evalemos el desempeñeo del modelo
print('Exactitud: ', accuracy_score(y_test, y_pred))
print('UAC-ROC ', roc_auc_score(y_test, y_pred))
print('Reporte de clasificacion: \n', classification_report(y_test, y_pred))
