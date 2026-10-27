# Benchmark build_f1_csv.py

**Timestamp:** 2026-10-27T17:46:40
**Wall clock:** 711.60 s (11.86 min)
**HTTP total:** 544.55 s (76.5%)
**Sleep total:** 166.50 s (23.4%)
**Compute + overhead:** 0.55 s (0.1%)
**Requests HTTP:** 333
**Llamadas a sleep:** 333

## Requests por endpoint

| Endpoint | Count | Total (s) | Avg (ms) | Status codes |
|---|---:|---:|---:|---|
| pitstops | 66 | 132.60 | 2009 | 200:66 |
| qualifying | 66 | 108.97 | 1651 | 200:66 |
| results | 66 | 118.16 | 1790 | 200:66 |
| schedule | 69 | 18.20 | 264 | 200:69 |
| sprint | 66 | 166.62 | 2524 | 200:66 |

## Reintentos implícitos

- URLs únicas: **267**
- Requests totales: **333**
- URLs solicitadas >1 vez: **3**
- Reintentos implícitos (requests extra): **66**

URLs repetidas (ordenadas alfabéticamente para diff estable):

| URL | Count |
|---|---:|
| `https://api.jolpi.ca/ergast/f1/2024.json` | 25 |
| `https://api.jolpi.ca/ergast/f1/2025.json` | 25 |
| `https://api.jolpi.ca/ergast/f1/2026.json` | 19 |

## Timings por fase

| Fase | Tiempo (s) | % del total |
|---|---:|---:|
| build_dataframe | 711.60 | 100.0% |

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
| avg_pitstops_per_driver vacío | 0 |
| dnfs vacío | 0 |
| Suma dnfs | 539 |

## Detección de truncamiento de pitstops (bug del `.json`)

- Carreras chequeadas: **66**
- Con truncamiento: **0**
