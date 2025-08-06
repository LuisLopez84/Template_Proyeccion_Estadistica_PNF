import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pwlf

# 📂 Ruta base donde están las carpetas EscenarioN
# BASE_DIR = r"D:\Jmeter_Prueba_Pipeline"  # <-- CAMBIA ESTA RUTA
BASE_DIR = r"D:\Jmeter_Prueba_Pipeline_Varios\ConsultaProductoCapaSMP"

# 📂 Carpeta de salida para resultados
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Verificar que la ruta base existe
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"❌ La ruta especificada no existe: {BASE_DIR}")

# 📈 Función logística para regresión no lineal
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# 📊 Procesar archivo JTL
def process_jtl(file_path):
    df = pd.read_csv(file_path)
    avg_rps = len(df) / 60  # 60 seg por escenario
    p95_latency = np.percentile(df["elapsed"], 95)
    error_rate = len(df[df["success"] == False]) / len(df) * 100
    return avg_rps, p95_latency, error_rate

# 🔍 Buscar y procesar todos los JTL en carpetas EscenarioN
results = []
for folder in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder)
    if os.path.isdir(folder_path) and folder.lower().startswith("escenario"):
        for file in os.listdir(folder_path):
            if file.endswith(".jtl"):
                usuarios = int(''.join([c for c in file if c.isdigit()]))
                avg_rps, p95_latency, error_rate = process_jtl(os.path.join(folder_path, file))
                results.append([usuarios, avg_rps, p95_latency, error_rate])

if not results:
    raise FileNotFoundError("❌ No se encontraron archivos .jtl en las carpetas EscenarioN.")

# 📑 Guardar resultados consolidados
df_results = pd.DataFrame(results, columns=["Usuarios", "TPS", "P95 Latency (ms)", "Error %"])
df_results.to_csv(os.path.join(OUTPUT_DIR, "resumen.csv"), index=False)

# 📈 Regresión no lineal para RPS
xdata = df_results["Usuarios"]
ydata = df_results["TPS"]
popt, _ = curve_fit(logistic, xdata, ydata, p0=[max(ydata), 0.1, np.median(xdata)])
x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = logistic(x_fit, *popt)

# 📉 Cambio de pendiente
pwlf_model = pwlf.PiecewiseLinFit(xdata, ydata)
breaks = pwlf_model.fit(2)
breakpoint_x = breaks[1]

# 📊 Graficar resultados
plt.figure()
plt.scatter(xdata, ydata, label="Datos")
plt.plot(x_fit, y_fit, label="Regresión logística", color="red")

plt.axvline(breakpoint_x, color="green", linestyle="--", label=f"Breakpoint: {breakpoint_x:.1f} usuarios")
plt.xlabel("Usuarios (hilos)")
plt.ylabel("TPS")
plt.legend()
plt.title("Regresión no lineal y cambio de pendiente")
plt.savefig(os.path.join(OUTPUT_DIR, "regresion_tps.png"))
plt.close()

print("✅ Análisis completado.")
print(f"📂 Resultados guardados en: {os.path.abspath(OUTPUT_DIR)}")