# ----------------------------------------------------------------------------------------------------------------------
# Curva ROC con datos desbalanceados
# ----------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------------------------------------
# PUNTO 1
# ----------------------------------------------------------------------------------------------------------------------
# Objetivo: Ajustar el umbral de clasificación en el conjunto de datos balanceados/desbalanceado y observar los cambios
# en la matriz de confusión y la curva ROC.

# RESPONDE LAS SIGUIENTES PREGUNTAS

# 1. ¿Cómo cambia la matriz de confusión al ajustar el umbral de clasificación?
# 2. ¿Cómo afecta el ajuste del umbral a la curva ROC y al AUC-ROC?
# ----------------------------------------------------------------------------------------------------------------------
# 1. Bajar el umbral por ejemplo a 0.3, hace que el modelo clasifique más casos como Clase 1, subiendo los falsos
#    positivos y bajando los falsos negativos. Si se sube a un 0.7 o superior hace todo lo contratario.

# 2. La curva ROC y el AUC no cambian al ajustar el umbral. El AUC del desbalanceado es 0.88 y del balanceado 0.92, y
#    eso es fijo. Lo que sí cambia es en qué punto de esa curva operas, cada umbral corresponde a un par (FPR, TPR)
#    distinto sobre la misma curva.
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# generar dos clases desbalanceadas
# ----------------------------------------------------------------------------------------------------------------------

x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.8, 0.2], random_state=42)

# dividir nuestros datos para train y para test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# entrenamos
modelo = LogisticRegression(max_iter=500)
modelo.fit(x_train, y_train)

# realizar predicciones
y_pred = modelo.predict(x_test)
y_pred_prob = modelo.predict_proba(x_test)[:, 1]

# creamos matriz de confusion
mc = confusion_matrix(y_test, y_pred)

mcd = ConfusionMatrixDisplay(mc, display_labels=['Clase 0', 'Clase 1'])
mcd.plot()

# calcular la tasa de fpr, y tpr
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('Curva ROC (datos desbalanceados)')
plt.legend(loc='lower right')
plt.show()

print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------------------------------------------------------
# generar dos clases balanceadas
# ----------------------------------------------------------------------------------------------------------------------

x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.5, 0.5], random_state=42)

# dividir nuestros datos para train y para test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# entrenamos
modelo = LogisticRegression(max_iter=500)
modelo.fit(x_train, y_train)

# realizar predicciones
y_pred = modelo.predict(x_test)
y_pred_prob = modelo.predict_proba(x_test)[:, 1]

# creamos matriz de confusion
mc = confusion_matrix(y_test, y_pred)

mcd = ConfusionMatrixDisplay(mc)
mcd.plot()

# calcular la tasa de fpr, y tpr
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('Curva ROC (datos balanceados)')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# Regenerar el dataset

x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.8, 0.2], random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ----------------------------------------------------------------------------------------------------------------------
# PUNTO 2
# ----------------------------------------------------------------------------------------------------------------------
# Objetivo: Completa el código de arriba para predecir nuevo valores con sobremuestreo
# Calcula la matriz de confusión
# Calcula la curva ROC
# Calcula el área bajo la curva
# Grafica la curva ROC
# ----------------------------------------------------------------------------------------------------------------------
# NOTA: Usar clases desbalanceadas
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Manejo de clases desbalanceadas con técnicas de sobremuestreo y submuestreo
# ----------------------------------------------------------------------------------------------------------------------


# Aplicar SMOTE para sobremuestrear la clase minoritaria
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x_train, y_train)

# Entrenar el modelo con datos sobremuestreados
modelo_smote = LogisticRegression(max_iter=10000)
modelo_smote.fit(x_smote, y_smote)

# Predecir nuevos valores
y_pred_smote = modelo_smote.predict(x_test)
y_pred_prob_smote = modelo_smote.predict_proba(x_test)[:, 1]

# Matriz de confusión
mc_smote = confusion_matrix(y_test, y_pred_smote)
mcd_smote = ConfusionMatrixDisplay(mc_smote, display_labels=['Clase 0', 'Clase 1'])
mcd_smote.plot()
plt.title('Matriz de Confusión - SMOTE')
plt.show()

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_smote)
roc_auc = roc_auc_score(y_test, y_pred_prob_smote)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC - SMOTE (datos desbalanceados)')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# PUNTO 3
# ----------------------------------------------------------------------------------------------------------------------
# Objetivo: Completa el códigode arriba para predecir nuevo valores con submuestreo
# Calcula la matriz de confusión
# Calcula la curva ROC
# Calcula el área bajo la curva
# Grafica la curva ROC
# ----------------------------------------------------------------------------------------------------------------------
# NOTA: Usar clases desbalanceadas
# ----------------------------------------------------------------------------------------------------------------------


# Aplicar submuestreo para reducir la clase mayoritaria
rus = RandomUnderSampler(random_state=42)
x_rus, y_rus = rus.fit_resample(x_train, y_train)

# Entrenar el modelo con datos submuestreados
modelo_rus = LogisticRegression(max_iter=10000)
modelo_rus.fit(x_rus, y_rus)

# Predecir nuevos valores
y_pred_rus = modelo_rus.predict(x_test)
y_pred_prob_rus = modelo_rus.predict_proba(x_test)[:, 1]

# Matriz de confusión
mc_rus = confusion_matrix(y_test, y_pred_rus)
mcd_rus = ConfusionMatrixDisplay(mc_rus, display_labels=['Clase 0', 'Clase 1'])
mcd_rus.plot()
plt.title('Matriz de Confusión - RUS')
plt.show()

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rus)
roc_auc = roc_auc_score(y_test, y_pred_prob_rus)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC= {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC - RUS (datos desbalanceados)')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# PUNTO 4
# ----------------------------------------------------------------------------------------------------------------------
# RESPONDE LAS SIGUIENTES PREGUNTAS
# ----------------------------------------------------------------------------------------------------------------------
# ¿Cómo cambia la matriz de confusión al aplicar técnicas de sobremuestreo y submuestreo?
# ¿Qué técnica (SMOTE o submuestreo) proporciona un mejor rendimiento en términos de AUC-ROC?
# ¿Qué observaciones puedes hacer sobre el impacto del balanceo de clases en el rendimiento del modelo?
# ----------------------------------------------------------------------------------------------------------------------
# 1. Las dos tecnicas mueven la matriz en la misma dirección, bajan los FN y suben los FP respecto al modelo base. El
#    modelo base tenia 21 FP y 9 FN, por el ado del modelo de SMOTE se tiene 16 16 FP con 11 FN y el modelo de RUS tiene
#    23 FP con 13 FN.
#    Esto significa que el modelo que el modelo se hace mas agresivo detectando las clases minoritarias a consto de
#    Disminuir la precision de la clase mayoritaria.

# 2. Tanto SMOTE como RUS son muy similares el valor del curva de SMOTE es de 0.89 mientras que la de RUS es de 0.88,
#    una diferencia insignificante.

# 3. Balancear las clases no mejora el AUC de forma dramática porque el AUC ya captura el rendimiento global del
#    ranking. Donde sí impacta es en cómo el modelo distribuye los errores: sin balanceo el modelo prefiere acertar en
#    la clase mayoritaria y se pierde muchos casos de Clase 1.
# ----------------------------------------------------------------------------------------------------------------------
