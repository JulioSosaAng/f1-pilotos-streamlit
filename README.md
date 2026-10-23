# 🏎️ F1 Data Analysis & Dashboard: 2024 – 2026

Pipeline de datos end-to-end (ETL) y dashboard interactivo para el análisis comparativo de pilotos de Fórmula 1, abarcando las temporadas 2024, 2025 y la temporada en curso 2026.

🔗 **Ver App en vivo:** [f1-pilotos-2024-2025.streamlit.app](https://f1-pilotos-2024-2025.streamlit.app/)

---

## 🎯 Sobre el proyecto

El objetivo de este proyecto es responder a la pregunta: **¿Quién domina y cómo evolucionan los pilotos a lo largo de las temporadas?** 

Construí un flujo de trabajo completo en Python que extrae los datos directamente de la API pública de Jolpica (con respaldo en Ergast), procesa la información, calcula métricas clave y las expone en una aplicación web interactiva.

El pipeline está diseñado para ser **tolerante a fallos** (con reintentos automáticos) y lo suficientemente inteligente como para ignorar automáticamente las carreras futuras de la temporada en curso (2026) hasta que se disputen.

---

## ⚙️ Características Principales

### 1. Pipeline de Extracción (ETL) - `build_f1_csv.py`
- **Ingesta Robusta:** Consumo de la API de Jolpica (sucesora de Ergast) con sistema de reintentos automáticos y pausas progresivas para evitar bloqueos.
- **Procesamiento:** Extracción de datos de clasificación (Pole), resultados de carrera (Podio), vueltas rápidas (Fastest Laps) y conteo de abandonos (DNFs).
- **Filtro Dinámico:** Detección automática de fechas para omitir carreras futuras en la temporada 2026.
- **Cálculos:** Promedio de paradas en boxes (Pitstops) por piloto y detección de fines de semana con formato Sprint.
- **Exportación:** Generación de un dataset unificado en formato CSV listo para su análisis.

### 2. Dashboard Interactivo (Streamlit + Plotly) - `streamlit_app.py`
La aplicación permite filtrar por temporada y por piloto para analizar:
- **KPIs Generales:** Total de carreras, piloto con más victorias, porcentaje global de conversión Pole → Victoria y piloto con más vueltas rápidas.
- **Victorias por Piloto:** Gráfico de barras agrupado por temporada.
- **Poles Totales:** Gráfico de torta (Pie chart) para visualizar el dominio en clasificación.
- **Vueltas Rápidas:** Gráfico de barras horizontales detallando qué piloto se llevó la vuelta rápida en cada Gran Premio.
- **Conversión Pole → Victoria:** Gráfico de barras horizontales mostrando la eficiencia promedio de cada piloto al largar primero, acompañado de una tabla detallada con estadísticas resumidas.

### 3. Testing de API - `test_API.py`
Un script auxiliar para verificar el estado de los datos en la API, permitiendo visualizar rápidamente qué rondas de la temporada actual ya tienen resultados oficiales.

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3
- **Extracción de Datos:** `requests` (API de Jolpica / Ergast)
- **Manipulación de Datos:** Pandas, NumPy
- **Visualización:** Plotly Express
- **Frontend / Dashboard:** Streamlit

---

## 📂 Estructura del Proyecto

```text
f1-analysis/
├── build_f1_csv.py       # Script de ETL: descarga y limpia los datos de la API
├── streamlit_app.py      # Código de la aplicación web y los gráficos
├── test_API.py           # Script para testear la disponibilidad de datos de 2026
├── Data/
│   └── f1_2024_2025_2026_filled.csv  # Dataset generado por el ETL
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

---

## 🚀 Cómo ejecutar el proyecto localmente

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/JulioSosaAng/f1-pilotos.git
cd f1-pilotos
pip install -r requirements.txt
```

### 2. Actualizar los datos
Si querés volver a descargar los datos más recientes (incluyendo las últimas carreras disputadas de 2026), ejecutá el script de extracción:
```bash
python build_f1_csv.py
```

### 3. Ejecutar el Dashboard
Iniciá la aplicación de Streamlit para visualizar los gráficos:
```bash
streamlit run streamlit_app.py
```

---

## 📬 Contacto

- **GitHub:** [@JulioSosaAng](https://github.com/JulioSosaAng)
- **LinkedIn:** [jsangeles](https://linkedin.com/in/jsangeles)
- **Email:** julio.sosa.angeles@outlook.com.ar

---
<p align="center">
  <sub>Desarrollado por Julio Sosa Angeles</sub>
</p>