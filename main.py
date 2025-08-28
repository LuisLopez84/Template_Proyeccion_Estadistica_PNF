import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import curve_fit
from datetime import datetime

# ReportLab para PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Carpeta de salida centralizada
# ----------------------------
OUTPUT_DIR = os.path.join(os.environ.get("GITHUB_WORKSPACE", os.getcwd()), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Función logística
# ----------------------------
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# ----------------------------
# Ruta base de escenarios
# ----------------------------
if len(sys.argv) > 1:
    escenarios_dir = sys.argv[1]
else:
    escenarios_dir = r"D:\Jmeter_Prueba_Pipeline_Varios\ConsultaProductoCapaSMP"

usuarios = []
tps_promedio = []
resultados_escenarios = []

# ----------------------------
# Leer datos de todos los .jtl en cada carpeta EscenarioN
# ----------------------------
for carpeta in sorted(os.listdir(escenarios_dir)):
    if carpeta.lower().startswith("escenario"):
        try:
            hilos = int(''.join(filter(str.isdigit, carpeta)))
            carpeta_path = os.path.join(escenarios_dir, carpeta)
            if not os.path.isdir(carpeta_path):
                continue

            jtl_files = [f for f in os.listdir(carpeta_path) if f.lower().endswith(".jtl")]
            if not jtl_files:
                continue

            total_registros = 0
            total_tiempo = 0
            tiempos_respuesta = []
            errores = 0

            for jtl in jtl_files:
                archivo_jtl = os.path.join(carpeta_path, jtl)
                df = pd.read_csv(archivo_jtl)

                if "timeStamp" not in df.columns or "elapsed" not in df.columns:
                    continue

                tiempo_total_seg = (df["timeStamp"].max() - df["timeStamp"].min()) / 1000
                if tiempo_total_seg > 0:
                    total_registros += len(df)
                    total_tiempo += tiempo_total_seg
                    tiempos_respuesta.extend(df["elapsed"].tolist())
                    if "success" in df.columns:
                        errores += len(df[df["success"] == False])

            if total_tiempo > 0 and total_registros > 0:
                tps = total_registros / total_tiempo
                avg_resp = np.mean(tiempos_respuesta)
                min_resp = np.min(tiempos_respuesta)
                max_resp = np.max(tiempos_respuesta)
                error_pct = (errores / total_registros) * 100 if total_registros > 0 else 0

                usuarios.append(hilos)
                tps_promedio.append(tps)

                resultados_escenarios.append({
                    "escenario": carpeta,
                    "usuarios": hilos,
                    "transacciones": total_registros,
                    "tiempo_prom": avg_resp,
                    "tiempo_min": min_resp,
                    "tiempo_max": max_resp,
                    "error_pct": error_pct,
                    "tps": tps
                })

        except Exception as e:
            print(f"Error procesando {carpeta}: {e}")

usuarios = np.array(usuarios)
tps_promedio = np.array(tps_promedio)

# Ordenar datos
orden = np.argsort(usuarios)
usuarios = usuarios[orden]
tps_promedio = tps_promedio[orden]

df_escenarios = pd.DataFrame(resultados_escenarios)

# ----------------------------
# Ajuste de curva logística
# ----------------------------
tiene_ajuste = False
L = k = x0 = breakpoint = None
y_pred = []

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
# Generar gráfica en OUTPUT_DIR
# ----------------------------
grafica_path = os.path.join(OUTPUT_DIR, "grafica_regresion.png")

plt.figure(figsize=(10, 6))
plt.scatter(usuarios, tps_promedio, label="Datos reales", color="blue")

if tiene_ajuste:
    x_fit = np.linspace(min(usuarios), max(usuarios), 200)
    y_fit = logistic(x_fit, L, k, x0)
    plt.plot(x_fit, y_fit, color="red", linewidth=2, label="Regresión logística")
    plt.axvline(breakpoint, color="green", linestyle="--", label=f"Breakpoint: {breakpoint:.1f} usuarios")

plt.title("Regresión no lineal - TPS vs Usuarios")
plt.xlabel("Usuarios (hilos)")
plt.ylabel("TPS")
plt.legend()
plt.grid(True)
plt.savefig(grafica_path)
plt.close()

# ----------------------------
# Guardar resultados CSV en OUTPUT_DIR
# ----------------------------
csv_path = os.path.join(OUTPUT_DIR, "resultados_grafana.csv")
df_resultados = pd.DataFrame({
    "usuarios": usuarios,
    "tps_promedio": tps_promedio
})
df_resultados.to_csv(csv_path, index=False)

# ----------------------------
# Calcular totales
# ----------------------------
if not df_escenarios.empty:
    total_transacciones = df_escenarios["transacciones"].sum()
    total_tiempo_prom = df_escenarios["tiempo_prom"].mean()
    total_tiempo_min = df_escenarios["tiempo_min"].mean()
    total_tiempo_max = df_escenarios["tiempo_max"].mean()
    total_error_pct = df_escenarios["error_pct"].mean()
    total_tps = df_escenarios["tps"].mean()
else:
    total_transacciones = total_tiempo_prom = total_tiempo_min = 0
    total_tiempo_max = total_error_pct = total_tps = 0

# ----------------------------
# Generar PDF en OUTPUT_DIR
# ----------------------------
pdf_path = os.path.join(OUTPUT_DIR, "Informe_Proyeccion_Estadistica_PNF.pdf")
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    topMargin=50,      # margen superior en puntos (72 = 1 pulgada)
    leftMargin=50,     # margen izquierdo
    rightMargin=50,    # margen derecho
    bottomMargin=50    # margen inferior
)
styles = getSampleStyleSheet()
elements = []


# Título
elements.append(Paragraph("<b>PROYECCIÓN ESTADÍSTICA DEL COMPORTAMIENTO DEL SERVICIO</b>", styles["Title"]))
elements.append(Spacer(1, 6))
elements.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]))
elements.append(Spacer(1, 20))



# Insertar gráfica
elements.append(Paragraph("<b>Gráfica de Regresión no lineal</b>", styles["Heading2"]))
elements.append(Spacer(1, 12))
elements.append(Image(grafica_path, width=450, height=300))
elements.append(Spacer(1, 20))

# Resultados clave
elements.append(Paragraph("<b>RESULTADOS CLAVE:</b>", styles["Heading2"]))
if tiene_ajuste:
    elements.append(Paragraph(f"• Breakpoint exacto: <b>{breakpoint:.2f}</b> usuarios/hilos", styles["Normal"]))
    elements.append(Paragraph(f"• Capacidad máxima estimada (L): <b>{L:.2f}</b> TPS", styles["Normal"]))
    elements.append(Paragraph(f"• Recomendación: No superar <b>{breakpoint:.0f}</b> hilos para mantener eficiencia", styles["Normal"]))
else:
    elements.append(Paragraph("No se pudo realizar la regresión logística por datos insuficientes", styles["Normal"]))
elements.append(Spacer(1, 15))

# Explicación fórmula (debajo de la gráfica y en la misma hoja)
elements.append(Paragraph("<b>FÓRMULA UTILIZADA:</b>", styles["Heading2"]))
elements.append(Paragraph("<font face='Courier'>def logistic(x, L, k, x0):</font>", styles["Normal"]))
elements.append(Paragraph("<font face='Courier'>    return L / (1 + np.exp(-k * (x - x0)))</font>", styles["Normal"]))

formula_desc = [
    "x → Número de usuarios/hilos",
    "L → Capacidad máxima (TPS)",
    "k ® Velocidad de crecimiento de la curva",
    "x0 ® Punto medio (usuarios donde la curva alcanza la mitad de L)"
]
for line in formula_desc:
    elements.append(Paragraph(f"• {line}", styles["Normal"]))

elements.append(Spacer(1, 20))


# Tabla de resultados por escenario
if not df_escenarios.empty:
    tabla_data = [["Escenario", "Tx Enviadas", "(t)Resp Prom (ms)", "(t)Resp Min (ms)", "(t)Resp Max (ms)", "% Error", "TPS"]]
    for _, row in df_escenarios.iterrows():
        tabla_data.append([
            row["escenario"], int(row["transacciones"]),
            f"{row['tiempo_prom']:.2f}", f"{row['tiempo_min']:.2f}",
            f"{row['tiempo_max']:.2f}", f"{row['error_pct']:.2f}%",
            f"{row['tps']:.2f}"
        ])

    tabla = Table(tabla_data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))
    elements.append(Paragraph("<b>Resultados por Escenario</b>", styles["Heading2"]))
    elements.append(tabla)
    elements.append(Spacer(1, 20))

# Tabla resumen total
tabla_total = [
    ["Total Tx", "(t)Resp Prom (ms)", "(t)Prom Resp Min (ms)", "(t)Prom Resp Max (ms)", "Prom % Error", "Prom TPS"],
    [f"{total_transacciones}", f"{total_tiempo_prom:.2f}", f"{total_tiempo_min:.2f}",
     f"{total_tiempo_max:.2f}", f"{total_error_pct:.2f}%", f"{total_tps:.2f}"]
]
tabla2 = Table(tabla_total, repeatRows=1)
tabla2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
    ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
]))
elements.append(Paragraph("<b>Resumen General</b>", styles["Heading2"]))
elements.append(tabla2)

# Generar PDF
doc.build(elements)

print(f"[OK] Archivos generados en: {OUTPUT_DIR}")
print(f"- CSV: {csv_path}")
print(f"- PNG: {grafica_path}")
print(f"- PDF: {pdf_path}")