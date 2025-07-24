import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ✅ Crear subcarpeta 'output' dentro de la carpeta de este script
SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Cargar el dataset desde la ruta local
csv_path = os.path.join(SCRIPT_DIR, "Bike-Sharing-Dataset", "hour.csv")
bike_data = pd.read_csv(csv_path)

# 📊 Exploración de datos
print(bike_data.info())
print(bike_data.describe())
print(bike_data["season"].value_counts())

bike_data.hist(bins=50, figsize=(20, 15))
plt.suptitle("Distribuciones de las variables", fontsize=16)
plt.savefig(os.path.join(OUTPUT_DIR, "01_histograma_variables.png"))
plt.show()

# 🛠️ Ingeniería de características
bike_data["day_type"] = bike_data.apply(
    lambda row: "working day" if row["workingday"] == 1 else "holiday/weekend",
    axis=1
)
bike_data["day_type"].value_counts().plot(kind="bar", color="purple")
plt.title("Distribución de tipos de día")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_distribucion_tipo_dia.png"))
plt.show()

# 📈 Correlaciones
scatter_matrix(bike_data[["cnt", "temp", "hum", "windspeed"]], figsize=(12, 8))
plt.suptitle("Matriz de dispersión", fontsize=16)
plt.savefig(os.path.join(OUTPUT_DIR, "03_matriz_dispersion.png"))
plt.show()

bike_data.plot(kind="scatter", x="temp", y="cnt", alpha=0.2)
plt.title("Relación entre temperatura y cantidad de alquileres")
plt.savefig(os.path.join(OUTPUT_DIR, "04_temp_vs_cnt.png"))
plt.show()

corr_matrix = bike_data.corr()
print(corr_matrix["cnt"].sort_values(ascending=False))

# ✂️ Preparación de datos
X = bike_data.drop(["cnt", "casual", "registered", "day_type"], axis=1)
y = bike_data["cnt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Modelos
lin_reg = LinearRegression().fit(X_train, y_train)
tree_reg = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# 📏 Evaluación
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_reg.predict(X_test)))
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_reg.predict(X_test)))
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_reg.predict(X_test)))

# 🔁 Validación cruzada
lin_scores = np.sqrt(-cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10))
tree_scores = np.sqrt(-cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10))
forest_scores = np.sqrt(-cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10))

# 📋 Resultados
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "RMSE": [lin_rmse, tree_rmse, forest_rmse],
    "CrossVal Mean": [lin_scores.mean(), tree_scores.mean(), forest_scores.mean()],
    "CrossVal Std": [lin_scores.std(), tree_scores.std(), forest_scores.std()]
})
print(results)

# 📊 Gráfica comparativa de RMSE
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["RMSE"], color=["orange", "green", "blue"])
plt.ylabel("RMSE")
plt.title("Comparación de RMSE entre modelos")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_comparacion_rmse.png"))
plt.show()
