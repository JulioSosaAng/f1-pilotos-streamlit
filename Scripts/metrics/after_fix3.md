# Benchmark build_f1_csv.py

**Timestamp:** 2026-10-28T15:48:38
**Wall clock:** 625.97 s (10.43 min)
**HTTP total:** 396.48 s (63.3%)
**Sleep total:** 228.93 s (36.6%)
**Compute + overhead:** 0.56 s (0.1%)
**Requests HTTP:** 417
**Llamadas a sleep:** 150

## Requests por endpoint

| Endpoint | Count | Total (s) | Avg (ms) | Status codes |
|---|---:|---:|---:|---|
| pitstops | 110 | 114.96 | 1045 | 200:66, 429:44 |
| qualifying | 106 | 93.88 | 886 | 200:66, 429:40 |
| results | 98 | 99.26 | 1013 | 200:66, 429:32 |
| schedule | 3 | 2.20 | 734 | 200:3 |
| sprint | 100 | 86.18 | 862 | 200:66, 429:34 |

## Reintentos implícitos

- URLs únicas: **267**
- Requests totales: **417**
- URLs solicitadas >1 vez: **109**
- Reintentos implícitos (requests extra): **150**

URLs repetidas (ordenadas alfabéticamente para diff estable):

| URL | Count |
|---|---:|
| `https://api.jolpi.ca/ergast/f1/2025/10/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/10/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/10/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/10/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/11/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/11/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/12/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/12/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/13/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/14/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/14/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/14/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/15/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/15/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/15/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/17/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/18/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/19/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/19/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/2/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/2/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/20/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/21/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/21/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/21/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/23/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/23/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/24/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/24/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/3/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/3/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/3/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/4/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/4/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/5/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/5/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/5/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/6/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/6/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/6/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/7/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/7/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/7/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/8/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/8/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/8/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/8/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/1/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/1/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/10/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/11/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/11/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/11/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/12/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/13/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/15/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/16/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/16/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/17/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/17/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/18/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/2/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/3/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/3/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/4/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/4/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/5/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/5/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/6/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/6/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/7/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/7/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/7/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/9/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/9/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/results.json` | 2 |

## Timings por fase

| Fase | Tiempo (s) | % del total |
|---|---:|---:|
| build_dataframe | 625.95 | 100.0% |

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
