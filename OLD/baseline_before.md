# Benchmark build_f1_csv.py

**Timestamp:** 2026-10-27T16:49:55
**Wall clock:** 840.64 s (14.01 min)
**HTTP total:** 628.52 s (74.8%)
**Sleep total:** 211.50 s (25.2%)
**Compute + overhead:** 0.62 s (0.1%)
**Requests HTTP:** 374
**Llamadas a sleep:** 374

## Requests por endpoint

| Endpoint | Count | Total (s) | Avg (ms) | Status codes |
|---|---:|---:|---:|---|
| pitstops | 102 | 220.17 | 2159 | 200:59, 404:21, 429:22 |
| qualifying | 68 | 137.88 | 2028 | 200:66, 429:2 |
| results | 69 | 138.95 | 2014 | 200:66, 429:2 |
| schedule | 69 | 20.48 | 297 | 200:69 |
| sprint | 66 | 111.04 | 1682 | 200:66 |

## Reintentos implícitos

- URLs únicas: **274**
- Requests totales: **374**
- URLs solicitadas >1 vez: **22**
- Reintentos implícitos (requests extra): **100**

URLs repetidas (ordenadas alfabéticamente para diff estable):

| URL | Count |
|---|---:|
| `https://api.jolpi.ca/ergast/f1/2024.json` | 25 |
| `https://api.jolpi.ca/ergast/f1/2024/16/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025.json` | 25 |
| `https://api.jolpi.ca/ergast/f1/2026.json` | 19 |
| `https://api.jolpi.ca/ergast/f1/2026/10/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/13/pitstops?limit=2000.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/15/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/16/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/18/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/8/pitstops?limit=2000.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/8/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/10/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/14/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/15/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/16/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/18/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/8/pitstops?limit=2000.json` | 3 |
| `https://ergast.com/api/f1/2026/9/pitstops?limit=2000.json` | 3 |

## Timings por fase

| Fase | Tiempo (s) | % del total |
|---|---:|---:|
| build_dataframe | 840.62 | 100.0% |

## Dataframe resultante

| Métrica | Valor |
|---|---:|
| Filas totales | 66 |
| Filas temporada 2024 | 24 |
| Filas temporada 2025 | 24 |
| Filas temporada 2026 | 18 |
| winner_driver vacío | 0 |
| pole_driver vacío | 0 |
| fastest_lap_driver vacío | 0 |
| avg_pitstops_per_driver vacío | 7 |
| dnfs vacío | 0 |
| Suma dnfs | 539 |

## Detección de truncamiento de pitstops (bug del `.json`)

- Carreras chequeadas: **59**
- Con truncamiento: **30**

Carreras con datos truncados (primeras 20):

| URL | Devueltos | Total API | Truncado |
|---|---:|---:|:---:|
| `https://api.jolpi.ca/ergast/f1/2024/1/pitstops?limit=2000.json` | 30 | 43 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/10/pitstops?limit=2000.json` | 30 | 42 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/11/pitstops?limit=2000.json` | 30 | 45 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/12/pitstops?limit=2000.json` | 30 | 45 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/13/pitstops?limit=2000.json` | 30 | 40 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/14/pitstops?limit=2000.json` | 30 | 34 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/21/pitstops?limit=2000.json` | 30 | 35 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/22/pitstops?limit=2000.json` | 30 | 39 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/23/pitstops?limit=2000.json` | 30 | 60 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/3/pitstops?limit=2000.json` | 30 | 36 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/4/pitstops?limit=2000.json` | 30 | 54 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/5/pitstops?limit=2000.json` | 30 | 40 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2024/9/pitstops?limit=2000.json` | 30 | 42 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/1/pitstops?limit=2000.json` | 30 | 82 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/10/pitstops?limit=2000.json` | 30 | 82 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/11/pitstops?limit=2000.json` | 30 | 32 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/12/pitstops?limit=2000.json` | 30 | 35 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/15/pitstops?limit=2000.json` | 30 | 40 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/21/pitstops?limit=2000.json` | 30 | 37 | **SÍ** |
| `https://api.jolpi.ca/ergast/f1/2025/23/pitstops?limit=2000.json` | 30 | 41 | **SÍ** |

_(+10 carreras más truncadas)_
