import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, roc_curve, auc

# ======== CONFIGURACIÓN ========
data_dir = '/Users/melissasanchez/Desktop/IA/ArtificialIntelligence/Laboratorios/Lab1/Leaves'
output_dir = '/Users/melissasanchez/Desktop/IA/ArtificialIntelligence/Laboratorios/Lab1/'
img_size = (64, 64)

# ======== CARGA DE IMÁGENES Y ETIQUETADO MANUAL (BINARIA: 3p y 5p) ========
X = []
y = []
image_names = []

print("Cargando imágenes...")
files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
total_imgs = len(files)

for idx, file in enumerate(files):
    path = os.path.join(data_dir, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, img_size)
        X.append(resized.flatten())
        y.append('3p' if idx < total_imgs // 2 else '5p')
        image_names.append(file)

print("Total de imágenes cargadas:", len(X))
print("Clases asignadas como '3p' y '5p'.")

# ======== CONVERSIÓN A ARRAYS ========
X = np.array(X)
y = np.array(y)
image_names = np.array(image_names)

# ======== DIVISIÓN DE DATOS ========
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, image_names, test_size=0.3, random_state=42
)

# ======== ENTRENAMIENTO KNN ========
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ======== REPORTES Y MATRIZ DE CONFUSIÓN ========
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['3p', '5p']))

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['3p', '5p'], yticklabels=['3p', '5p'])
plt.title("Matriz de Confusión - KNN")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_confusion_knn.png"))
plt.close()

# ======== GRAFICAR PRECISIÓN VS K ========
accuracies = []
k_values = list(range(1, 10))
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)

plt.figure()
plt.plot(k_values, accuracies, marker='o', color='purple')
plt.title("Precisión vs Número de Vecinos (K)")
plt.xlabel("K")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_vs_k.png"))
plt.close()

best_k = k_values[np.argmax(accuracies)]
print(f"\n✅ Mejor valor de K: {best_k} con precisión: {max(accuracies):.2f}")

# ======== GUARDAR RESULTADOS EN CSV ========
df_resultados = pd.DataFrame({
    "Imagen": names_test,
    "Etiqueta Real": y_test,
    "Predicción": y_pred
})
df_resultados.to_csv(os.path.join(output_dir, "predicciones_knn.csv"), index=False)
print("✅ Archivo CSV guardado como 'predicciones_knn.csv'")

# ======== CLASIFICACIÓN BINARIA CON SGD Y CURVAS ========
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

print("\n=== Clasificador Binario MNIST ===")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_mnist, y_mnist = mnist["data"], mnist["target"].astype(np.uint8)

y_mnist_7 = (y_mnist == 7)

X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
    X_mnist, y_mnist_7, test_size=0.3, random_state=42
)

# ======== KNN ========
knn_mnist = KNeighborsClassifier()
knn_mnist.fit(X_train_mnist, y_train_mnist)
y_pred_knn = knn_mnist.predict(X_test_mnist)

# ======== Random Forest ========
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_mnist, y_train_mnist)
y_pred_rf = rf.predict(X_test_mnist)

# ======== EVALUACIÓN Y CURVAS ========
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n=== KNN ===")
print("Precisión:", precision_score(y_test_mnist, y_pred_knn))
print("Recall:", recall_score(y_test_mnist, y_pred_knn))
print("F1:", f1_score(y_test_mnist, y_pred_knn))

print("\n=== Random Forest ===")
print("Precisión:", precision_score(y_test_mnist, y_pred_rf))
print("Recall:", recall_score(y_test_mnist, y_pred_rf))
print("F1:", f1_score(y_test_mnist, y_pred_rf))

# ======== CURVAS CON RANDOM FOREST ========
y_scores_rf = rf.predict_proba(X_test_mnist)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test_mnist, y_scores_rf)

plt.plot(thresholds, precisions[:-1], label="Precisión")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.grid()
plt.title("Precisión y Recall vs Threshold")
plt.savefig(os.path.join(output_dir, "precision_recall_threshold.png"))
plt.close()

plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curva Precisión vs Recall")
plt.grid()
plt.savefig(os.path.join(output_dir, "precision_vs_recall.png"))
plt.close()

fpr, tpr, _ = roc_curve(y_test_mnist, y_scores_rf)
plt.plot(fpr, tpr)
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.grid()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

print("✅ Todas las gráficas y resultados han sido guardados en la carpeta Lab1.")
