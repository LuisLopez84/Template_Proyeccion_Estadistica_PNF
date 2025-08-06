import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# ----------------------------
# Función logística
# ----------------------------
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# ----------------------------
# Leer datos TPS desde carpetas EscenarioN
# ----------------------------
escenarios_dir = r"D:\Jmeter_Prueba_Pipeline_Varios\ConsultaProductoCapaSMP"  # Cambia esto
usuarios = []
tps_promedio = []

for carpeta in sorted(os.listdir(escenarios_dir)):
    if carpeta.lower().startswith("escenario"):
        try:
            hilos = int(''.join(filter(str.isdigit, carpeta)))
            archivo_jtl = os.path.join(escenarios_dir, carpeta, "resultados.jtl")

            if os.path.exists(archivo_jtl):
                df = pd.read_csv(archivo_jtl)
                if "timeStamp" not in df.columns:
                    print(f"El archivo {archivo_jtl} no tiene columna 'timeStamp'")
                    continue

                tiempo_total_seg = (df["timeStamp"].max() - df["timeStamp"].min()) / 1000
                if tiempo_total_seg <= 0:
                    print(f"Tiempo total inválido en {archivo_jtl}")
                    continue

                tps = len(df) / tiempo_total_seg
                usuarios.append(hilos)
                tps_promedio.append(tps)
        except Exception as e:
            print(f"Error procesando {carpeta}: {e}")

usuarios = np.array(usuarios)
tps_promedio = np.array(tps_promedio)

# Ordenar por número de hilos
orden = np.argsort(usuarios)
usuarios = usuarios[orden]
tps_promedio = tps_promedio[orden]

# ----------------------------
# Variables de control
# ----------------------------
tiene_ajuste = False
L = k = x0 = breakpoint = None
y_pred = []

# ----------------------------
# Intentar ajuste solo si hay datos suficientes
# ----------------------------
if len(usuarios) >= 3:
    try:
        p0 = [max(tps_promedio), 1, np.median(usuarios)]
        params, _ = curve_fit(logistic, usuarios, tps_promedio, p0, maxfev=10000)
        L, k, x0 = params
        breakpoint = x0
        y_pred = logistic(usuarios, L, k, x0)
        tiene_ajuste = True
    except Exception as e:
        print(f"No se pudo ajustar la curva: {e}")

# ----------------------------
# Graficar resultados
# ----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(usuarios, tps_promedio, label="Datos reales", color="blue")

if tiene_ajuste:
    x_fit = np.linspace(min(usuarios), max(usuarios), 200)
    y_fit = logistic(x_fit, L, k, x0)
    plt.plot(x_fit, y_fit, label="Regresión logística", color="red", linewidth=2)
    plt.axvline(breakpoint, color="green", linestyle="--", label=f"Breakpoint: {breakpoint:.1f} usuarios")
else:
    plt.text(0.5, 0.9, "No se pudo ajustar la curva\n(Datos insuficientes)",
             transform=plt.gca().transAxes, fontsize=12, color="red", ha="center")

plt.title("Comportamiento TPS vs Usuarios")
plt.xlabel("Usuarios (hilos)")
plt.ylabel("TPS")
plt.legend()
plt.grid(True)

grafica_path = os.path.join(escenarios_dir, "grafica_regresion.png")
plt.savefig(grafica_path)
plt.close()

# ----------------------------
# Generar informe PDF multipágina
# ----------------------------
pdf_path = os.path.join(escenarios_dir, "informe_regresion.pdf")
c = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter

# -------- Página 1: Resumen --------
c.setFont("Helvetica-Bold", 16)
c.drawString(50, height - 50, "Proyección Estadística del comportamiento del servicio.")

if tiene_ajuste:
    analisis = f"""
    Breakpoint: {breakpoint:.2f} usuarios/hilos
    Capacidad máxima estimada (L): {L:.2f} TPS
    Recomendación: No superar {breakpoint:.0f} hilos para mantener eficiencia.
    """
else:
    analisis = """
    No se pudo realizar la regresión logística debido a datos insuficientes.
    Requiere al menos 3 puntos para el ajuste.
    """

c.setFont("Helvetica", 10)
y_pos = height - 80
for linea in analisis.strip().split("\n"):
    c.drawString(50, y_pos, linea)
    y_pos -= 12

c.showPage()

# -------- Página 2: Tabla de datos --------
data = [["Usuarios", "TPS Real"] + (["TPS Estimado"] if tiene_ajuste else [])]
for i, u in enumerate(usuarios):
    fila = [int(u), f"{tps_promedio[i]:.2f}"]
    if tiene_ajuste:
        fila.append(f"{y_pred[i]:.2f}")
    data.append(fila)

tabla = Table(data, colWidths=[150] * len(data[0]))
tabla.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ("GRID", (0, 0), (-1, -1), 1, colors.black)
]))
tabla.wrapOn(c, width, height)
tabla.drawOn(c, 50, height - 300)

c.showPage()

# -------- Página 3: Imagen de la gráfica --------
c.drawImage(grafica_path, 50, 150, width=500, height=400)

c.save()
print(f"Informe multipágina generado en: {pdf_path}")