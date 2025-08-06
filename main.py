import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pwlf

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def process_jtl(file_path):
    df = pd.read_csv(file_path)
    avg_rps = len(df) / 60  # asumiendo 60 seg por escenario
    p95_latency = np.percentile(df["elapsed"], 95)
    error_rate = len(df[df["success"] == False]) / len(df) * 100
    return avg_rps, p95_latency, error_rate

# Consolidar datos
results = []
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".jtl"):
        usuarios = int(''.join([c for c in file if c.isdigit()]))
        avg_rps, p95_latency, error_rate = process_jtl(os.path.join(DATA_DIR, file))
        results.append([usuarios, avg_rps, p95_latency, error_rate])

df_results = pd.DataFrame(results, columns=["Usuarios", "RPS", "P95 Latency (ms)", "Error %"])
df_results.to_csv(os.path.join(OUTPUT_DIR, "resumen.csv"), index=False)

# Regresión no lineal para RPS
xdata = df_results["Usuarios"]
ydata = df_results["RPS"]
popt, _ = curve_fit(logistic, xdata, ydata, p0=[max(ydata), 0.1, np.median(xdata)])
x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = logistic(x_fit, *popt)

# Cambio de pendiente (breakpoint)
pwlf_model = pwlf.PiecewiseLinFit(xdata, ydata)
breaks = pwlf_model.fit(2)  # 2 segmentos
breakpoint_x = breaks[1]

# Graficar
plt.figure()
plt.scatter(xdata, ydata, label="Datos")
plt.plot(x_fit, y_fit, label="Regresión logística", color="red")
plt.axvline(breakpoint_x, color="green", linestyle="--", label=f"Breakpoint: {breakpoint_x:.1f} usuarios")
plt.xlabel("Usuarios (hilos)")
plt.ylabel("RPS")
plt.legend()
plt.title("Regresión no lineal y cambio de pendiente")
plt.savefig(os.path.join(OUTPUT_DIR, "regresion_rps.png"))
plt.close()

print("✅ Análisis completado. Resultados en carpeta 'output/'.")
