# Benchmark build_f1_csv.py

**Timestamp:** 2026-10-30T10:36:53
**Wall clock:** 577.19 s (9.62 min)
**HTTP total:** 301.07 s (52.2%)
**Sleep total:** 275.48 s (47.7%)
**Compute + overhead:** 0.64 s (0.1%)
**Requests HTTP:** 431
**Llamadas a sleep:** 176

## Requests por endpoint

| Endpoint | Count | Total (s) | Avg (ms) | Status codes |
|---|---:|---:|---:|---|
| pitstops | 107 | 72.51 | 678 | 200:63, 429:44 |
| qualifying | 110 | 97.79 | 889 | 200:63, 429:47 |
| results | 104 | 66.09 | 635 | 200:63, 429:41 |
| schedule | 5 | 2.26 | 452 | 200:3, 429:2 |
| sprint | 105 | 62.43 | 595 | 200:63, 429:42 |

## Reintentos implícitos

- URLs únicas: **255**
- Requests totales: **431**
- URLs solicitadas >1 vez: **122**
- Reintentos implícitos (requests extra): **176**

URLs repetidas (ordenadas alfabéticamente para diff estable):

| URL | Count |
|---|---:|
| `https://api.jolpi.ca/ergast/f1/2025.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/1/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/1/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/1/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/1/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/10/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/10/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/10/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/11/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/12/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/12/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/13/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/14/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/14/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/15/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/15/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/15/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/16/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/17/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/18/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/19/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/2/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/2/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/2/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/20/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/20/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/20/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/21/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/21/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/22/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/22/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/23/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/23/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/23/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/24/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/24/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/3/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/3/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/3/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/3/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/4/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/4/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/4/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/4/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/5/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/9/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/9/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/9/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/1/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/1/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/1/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/1/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/10/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/10/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/11/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/11/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/11/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/13/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/13/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/15/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/15/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/16/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/16/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/16/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/16/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/18/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/18/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/18/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/18/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/2/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/2/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/3/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/3/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/3/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/4/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/4/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/5/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/5/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/5/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/6/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/6/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/7/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/7/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/8/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/8/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/9/sprint.json` | 3 |

## Timings por fase

| Fase | Tiempo (s) | % del total |
|---|---:|---:|
| build_dataframe | 577.18 | 100.0% |

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

- Carreras chequeadas: **63**
- Con truncamiento: **0**
