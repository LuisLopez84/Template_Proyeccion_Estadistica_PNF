import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pandas as pd
from scipy.optimize import curve_fit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# ----------------------------
# Función logística
# ----------------------------
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# ----------------------------
# Ruta base de escenarios
# ----------------------------
# Leer argumento desde línea de comandos
if len(sys.argv) > 1:
    escenarios_dir = sys.argv[1]
else:
    escenarios_dir = r"D:\Jmeter_Prueba_Pipeline_Varios\ConsultaProductoCapaSMP"
usuarios = []
tps_promedio = []

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

            for jtl in jtl_files:
                archivo_jtl = os.path.join(carpeta_path, jtl)
                df = pd.read_csv(archivo_jtl)

                if "timeStamp" not in df.columns:
                    continue

                tiempo_total_seg = (df["timeStamp"].max() - df["timeStamp"].min()) / 1000
                if tiempo_total_seg > 0:
                    total_registros += len(df)
                    total_tiempo += tiempo_total_seg

            if total_tiempo > 0:
                tps = total_registros / total_tiempo
                usuarios.append(hilos)
                tps_promedio.append(tps)

        except Exception as e:
            print(f"Error procesando {carpeta}: {e}")

usuarios = np.array(usuarios)
tps_promedio = np.array(tps_promedio)

# Ordenar datos
orden = np.argsort(usuarios)
usuarios = usuarios[orden]
tps_promedio = tps_promedio[orden]

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
# Generar gráfica
# ----------------------------
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

grafica_path = os.path.join(escenarios_dir, "grafica_regresion.png")
plt.savefig(grafica_path)
plt.close()


# después de calcular `usuarios` y `tps_promedio`
df_resultados = pd.DataFrame({
    "usuarios": usuarios,
    "tps_promedio": tps_promedio
})

# Guardar en CSV para Grafana (puede estar en carpeta compartida)
df_resultados.to_csv("D:/jmeter_regresion_analysis/output/resultados_grafana.csv", index=False)


# ----------------------------
# Generar PDF con formato personalizado
# ----------------------------
pdf_path = os.path.join(escenarios_dir, "Informe_Proyeccion_Estadistica_PNF.pdf")
c = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter

# Título
c.setFont("Helvetica-Bold", 18)
c.drawString(165, height - 35, "PROYECCIÓN ESTADÍSTICA DEL")
c.drawString(155, height - 55, "COMPORTAMIENTO DEL SERVICIO")
c.setFont("Helvetica", 7)
c.drawString(50, height - 75, f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# Calcular posición Y para la gráfica
posicion_y_grafica = height - 80 - 350  # 400 es la altura de la gráfica

# Dibujar la gráfica debajo del texto
c.drawImage(grafica_path, 50, posicion_y_grafica, width=450, height=350)


# Resultados clave
c.setFont("Helvetica-Bold", 12)
c.drawString(50, height - 450, "RESULTADOS CLAVE:")
c.setFont("Helvetica", 10)
if tiene_ajuste:
    c.drawString(60, height - 465, f"• Breakpoint exacto: {breakpoint:.2f} usuarios/hilos")
    c.drawString(60, height - 480, f"• Capacidad máxima estimada (L): {L:.2f} TPS")
    c.drawString(60, height - 495, f"• Recomendación: No superar {breakpoint:.0f} hilos para mantener eficiencia")
else:
    c.drawString(60, height - 465, "No se pudo realizar la regresión logística por datos insuficientes")

# Explicación fórmula
c.setFont("Helvetica-Bold", 12)
c.drawString(50, height - 530, "FÓRMULA UTILIZADA:")
c.setFont("Courier", 9)
c.drawString(60, height - 545, "def logistic(x, L, k, x0):")
c.drawString(60, height - 560, "    return L / (1 + np.exp(-k * (x - x0)))")

c.setFont("Helvetica", 9)
formula_desc = [
    "x → Número de usuarios/hilos",
    "L → Capacidad máxima (TPS)",
    "k → Velocidad de crecimiento de la curva",
    "x0 → Punto medio (usuarios donde la curva alcanza la mitad de L)"
]
y_pos = height - 580
for line in formula_desc:
    c.drawString(60, y_pos, f"• {line}")
    y_pos -= 12

# Interpretación gráfica
c.setFont("Helvetica-Bold", 12)
c.drawString(50, y_pos - 10, "INTERPRETACIÓN DE LA GRÁFICA:")
c.setFont("Helvetica", 9)
interpretacion = [
    "1. Eje X: Usuarios/hilos concurrentes",
    "2. Eje Y: TPS (Transacciones por segundo)",
    "3. Puntos azules: Datos reales de las pruebas",
    "4. Línea roja: Regresión logística (curva ajustada)",
    "5. Línea verde: Breakpoint (punto de saturación)"
]
y_pos -= 25
for line in interpretacion:
    c.drawString(60, y_pos, f"{line}")
    y_pos -= 12

c.setFont("Helvetica", 7)
interpretacion = [
    "1. Estrategia estadística aplicada:",
    "a. Análisis de regresión no lineal",
    "2. Método de análisis estadístico aplicado:",
    "a. Regresión no lineal (modelo logístico o exponencial)",
    "b. Cambio de pendiente (Análisis de punto de quiebre / breakpoint))",
]
y_pos -= 25
for line in interpretacion:
    c.drawString(60, y_pos, f"{line}")
    y_pos -= 12

c.showPage()

# Segunda página: gráfica
#c.drawImage(grafica_path, 50, 200, width=500, height=400)
#c.showPage()

c.save()
print(f"Informe generado en: {pdf_path}")