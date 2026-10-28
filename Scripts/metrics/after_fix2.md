# Benchmark build_f1_csv.py

**Timestamp:** 2026-10-28T15:12:50
**Wall clock:** 623.30 s (10.39 min)
**HTTP total:** 364.12 s (58.4%)
**Sleep total:** 258.50 s (41.5%)
**Compute + overhead:** 0.69 s (0.1%)
**Requests HTTP:** 381
**Llamadas a sleep:** 381

## Requests por endpoint

| Endpoint | Count | Total (s) | Avg (ms) | Status codes |
|---|---:|---:|---:|---|
| pitstops | 103 | 90.89 | 882 | 200:66, 429:37 |
| qualifying | 88 | 94.96 | 1079 | 200:66, 429:22 |
| results | 94 | 85.06 | 905 | 200:66, 429:28 |
| schedule | 3 | 4.29 | 1429 | 200:3 |
| sprint | 93 | 88.92 | 956 | 200:66, 429:27 |

## Reintentos implícitos

- URLs únicas: **267**
- Requests totales: **381**
- URLs solicitadas >1 vez: **92**
- Reintentos implícitos (requests extra): **114**

URLs repetidas (ordenadas alfabéticamente para diff estable):

| URL | Count |
|---|---:|
| `https://api.jolpi.ca/ergast/f1/2025/10/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/10/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/11/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/11/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/12/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/13/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/13/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/13/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/14/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/14/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/14/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/15/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/16/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/17/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/18/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/18/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/19/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/20/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/21/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/21/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/22/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/23/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/23/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/23/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/24/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/7/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/8/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/8/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/9/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2025/9/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2025/9/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/10/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/11/qualifying.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/11/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/12/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/13/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/13/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/14/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/14/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/15/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/15/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/16/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/16/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/17/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/17/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/17/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/18/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/2/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/2/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/2/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/3/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/3/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/3/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/4/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/5/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/5/results.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/5/sprint.json` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/6/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/6/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/6/sprint.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/7/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/7/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/7/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/pitstops.json?limit=2000` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/qualifying.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/8/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/pitstops.json?limit=2000` | 3 |
| `https://api.jolpi.ca/ergast/f1/2026/9/results.json` | 2 |
| `https://api.jolpi.ca/ergast/f1/2026/9/sprint.json` | 3 |

## Timings por fase

| Fase | Tiempo (s) | % del total |
|---|---:|---:|
| build_dataframe | 623.29 | 100.0% |

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
