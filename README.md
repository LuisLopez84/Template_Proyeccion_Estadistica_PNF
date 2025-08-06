# JMeter Regression Analysis Tool

Este proyecto en Python procesa archivos `.jtl` generados por JMeter, consolida métricas y realiza:
- Regresión no lineal (curva logística)
- Detección de punto de quiebre (cambio de pendiente)

## Requisitos
- Python 3.9+
- Librerías: pandas, matplotlib, scipy, pwlf

Instalación de librerías:
```bash
pip install pandas matplotlib scipy pwlf
```

## Estructura esperada
Coloca todos los `.jtl` (uno por escenario) en la carpeta `data/`.

Ejemplo de nombres de archivos:
```
escenario_10.jtl
escenario_20.jtl
escenario_30.jtl
escenario_40.jtl
escenario_50.jtl
```

## Uso en IntelliJ IDEA
1. Abre IntelliJ IDEA
2. Ve a **File > Open** y selecciona la carpeta de este proyecto
3. Abre el archivo `main.py`
4. Ejecuta el script desde IntelliJ IDEA (botón de play) o desde terminal:
```bash
python main.py
```

El script:
- Lee todos los `.jtl`
- Calcula KPIs por escenario
- Aplica regresión no lineal
- Detecta punto de quiebre
- Genera gráficos en la carpeta `output/`

