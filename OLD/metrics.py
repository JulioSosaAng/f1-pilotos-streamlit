# metrics.py
# -*- coding: utf-8 -*-
"""
Instrumentación no invasiva para medir rendimiento de extractores F1.

No modifica la lógica: intercepta requests.get y time.sleep, y acumula
contadores/timings. Pensado para correr el mismo script antes y después
de una refactorización y comparar los reportes markdown lado a lado.

Uso:
    from metrics import Metrics
    M = Metrics()
    M.patch()            # intercepta requests.get y time.sleep
    M.start()

    with M.phase("build_dataframe"):
        df = build_dataframe()

    M.stop()
    M.set_dataframe_info(df)
    M.write_markdown("metrics/baseline_before.md")
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


_ENDPOINT_PATTERNS = [
    (re.compile(r"/qualifying(?:\.json)?"), "qualifying"),
    (re.compile(r"/results(?:\.json)?"), "results"),
    (re.compile(r"/sprint(?:\.json)?"), "sprint"),
    (re.compile(r"/pitstops"), "pitstops"),
    (re.compile(r"/f1/\d{4}(?:\.json)?(?:\?|$)"), "schedule"),
]


def _classify(url: str) -> str:
    for pattern, name in _ENDPOINT_PATTERNS:
        if pattern.search(url):
            return name
    return "other"


class Metrics:
    def __init__(self) -> None:
        # Wall clock
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # HTTP
        self.endpoint_count: Dict[str, int] = defaultdict(int)
        self.endpoint_time: Dict[str, float] = defaultdict(float)
        self.endpoint_status: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.url_count: Dict[str, int] = defaultdict(int)
        self.http_total_time: float = 0.0
        self.http_total_count: int = 0

        # Sleeps (clave para explicar el "non-HTTP time")
        self.sleep_total: float = 0.0
        self.sleep_count: int = 0

        # Fases
        self.phase_times: Dict[str, float] = {}

        # Detector de truncamiento de pitstops
        self.pitstop_checks: List[Dict[str, Any]] = []

        # Dataframe
        self.df_info: Dict[str, Any] = {}

        # Estado de patching
        self._patched = False
        self._orig_get = None
        self._orig_sleep = None

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------
    def patch(self) -> None:
        if self._patched:
            return
        self._orig_get = requests.get
        self._orig_sleep = time.sleep
        original_get = self._orig_get
        original_sleep = self._orig_sleep
        m = self

        def patched_get(url, *args, **kwargs):
            ep = _classify(url)
            m.endpoint_count[ep] += 1
            m.url_count[url] += 1
            m.http_total_count += 1

            t0 = time.perf_counter()
            try:
                resp = original_get(url, *args, **kwargs)
                m.endpoint_status[ep][resp.status_code] += 1

                # Detector del bug del ".json" en pitstops
                if ep == "pitstops" and resp.status_code == 200:
                    try:
                        data = resp.json()
                        mr = data.get("MRData", {})
                        total = int(mr.get("total", 0))
                        races = mr.get("RaceTable", {}).get("Races", [])
                        returned = len(races[0].get("PitStops", [])) if races else 0
                        m.pitstop_checks.append({
                            "url": url,
                            "returned": returned,
                            "total": total,
                            "truncated": returned < total,
                        })
                    except Exception:
                        pass

                return resp
            finally:
                elapsed = time.perf_counter() - t0
                m.endpoint_time[ep] += elapsed
                m.http_total_time += elapsed

        def patched_sleep(seconds):
            try:
                m.sleep_total += float(seconds)
                m.sleep_count += 1
            except Exception:
                pass
            return original_sleep(seconds)

        requests.get = patched_get
        time.sleep = patched_sleep
        self._patched = True

    def unpatch(self) -> None:
        if self._patched:
            if self._orig_get is not None:
                requests.get = self._orig_get
            if self._orig_sleep is not None:
                time.sleep = self._orig_sleep
            self._patched = False

    # ------------------------------------------------------------------
    # Tiempo total y fases
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        self.end_time = time.perf_counter()

    @property
    def wall_clock(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phase_times[name] = self.phase_times.get(name, 0.0) + (
                time.perf_counter() - t0
            )

    # ------------------------------------------------------------------
    # Dataframe resultante
    # ------------------------------------------------------------------
    def set_dataframe_info(self, df: Optional[pd.DataFrame]) -> None:
        if df is None or df.empty:
            self.df_info = {"empty": True}
            return

        per_season = {}
        if "season" in df.columns:
            for season, count in df.groupby("season").size().items():
                try:
                    per_season[int(season)] = int(count)
                except Exception:
                    per_season[str(season)] = int(count)

        def _empty_count(col: str) -> int:
            if col not in df.columns:
                return -1
            s = df[col]
            return int(s.isna().sum() + (s.astype(str) == "").sum())

        def _sum_numeric(col: str) -> float:
            if col not in df.columns:
                return -1.0
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

        self.df_info = {
            "empty": False,
            "rows": len(df),
            "per_season": per_season,
            "empty_winner_driver": _empty_count("winner_driver"),
            "empty_pole_driver": _empty_count("pole_driver"),
            "empty_fastest_lap_driver": _empty_count("fastest_lap_driver"),
            "empty_avg_pitstops": _empty_count("avg_pitstops_per_driver"),
            "empty_dnfs": _empty_count("dnfs"),
            "sum_dnfs": _sum_numeric("dnfs"),
        }

    # ------------------------------------------------------------------
    # Reporte markdown
    # ------------------------------------------------------------------
    def to_markdown(self, title: str = "Benchmark") -> str:
        L: List[str] = []
        w = L.append

        wall = self.wall_clock
        http = self.http_total_time
        sleep_t = self.sleep_total
        compute_t = max(wall - http - sleep_t, 0.0)

        w(f"# {title}")
        w("")
        w(f"**Timestamp:** {datetime.now().isoformat(timespec='seconds')}")
        w(f"**Wall clock:** {wall:.2f} s ({wall/60:.2f} min)")
        w(f"**HTTP total:** {http:.2f} s ({(100*http/wall if wall else 0):.1f}%)")
        w(f"**Sleep total:** {sleep_t:.2f} s ({(100*sleep_t/wall if wall else 0):.1f}%)")
        w(f"**Compute + overhead:** {compute_t:.2f} s ({(100*compute_t/wall if wall else 0):.1f}%)")
        w(f"**Requests HTTP:** {self.http_total_count}")
        w(f"**Llamadas a sleep:** {self.sleep_count}")
        w("")

        # Requests por endpoint (orden alfabético para que el diff sea estable)
        w("## Requests por endpoint")
        w("")
        w("| Endpoint | Count | Total (s) | Avg (ms) | Status codes |")
        w("|---|---:|---:|---:|---|")
        for ep in sorted(self.endpoint_count.keys()):
            count = self.endpoint_count[ep]
            total = self.endpoint_time[ep]
            avg_ms = 1000 * total / count if count else 0
            codes = ", ".join(
                f"{code}:{n}" for code, n in sorted(self.endpoint_status[ep].items())
            )
            w(f"| {ep} | {count} | {total:.2f} | {avg_ms:.0f} | {codes} |")
        w("")

        # Retries implícitos
        repeated = {u: n for u, n in self.url_count.items() if n > 1}
        extra = sum(n - 1 for n in repeated.values())
        w("## Reintentos implícitos")
        w("")
        w(f"- URLs únicas: **{len(self.url_count)}**")
        w(f"- Requests totales: **{self.http_total_count}**")
        w(f"- URLs solicitadas >1 vez: **{len(repeated)}**")
        w(f"- Reintentos implícitos (requests extra): **{extra}**")
        w("")
        if repeated:
            w("URLs repetidas (ordenadas alfabéticamente para diff estable):")
            w("")
            w("| URL | Count |")
            w("|---|---:|")
            for url in sorted(repeated.keys()):
                w(f"| `{url}` | {repeated[url]} |")
            w("")

        # Fases
        if self.phase_times:
            w("## Timings por fase")
            w("")
            w("| Fase | Tiempo (s) | % del total |")
            w("|---|---:|---:|")
            for name in sorted(self.phase_times.keys()):
                t = self.phase_times[name]
                pct = 100 * t / wall if wall else 0
                w(f"| {name} | {t:.2f} | {pct:.1f}% |")
            w("")

        # Dataframe
        w("## Dataframe resultante")
        w("")
        if self.df_info.get("empty", True):
            w("_Dataframe vacío o no establecido._")
        else:
            w("| Métrica | Valor |")
            w("|---|---:|")
            w(f"| Filas totales | {self.df_info['rows']} |")
            for season in sorted(self.df_info.get("per_season", {}).keys()):
                w(f"| Filas temporada {season} | {self.df_info['per_season'][season]} |")
            w(f"| winner_driver vacío | {self.df_info['empty_winner_driver']} |")
            w(f"| pole_driver vacío | {self.df_info['empty_pole_driver']} |")
            w(f"| fastest_lap_driver vacío | {self.df_info['empty_fastest_lap_driver']} |")
            w(f"| avg_pitstops_per_driver vacío | {self.df_info['empty_avg_pitstops']} |")
            w(f"| dnfs vacío | {self.df_info['empty_dnfs']} |")
            w(f"| Suma dnfs | {self.df_info['sum_dnfs']:.0f} |")
        w("")

        # Truncamiento de pitstops
        if self.pitstop_checks:
            truncated = [c for c in self.pitstop_checks if c["truncated"]]
            w("## Detección de truncamiento de pitstops (bug del `.json`)")
            w("")
            w(f"- Carreras chequeadas: **{len(self.pitstop_checks)}**")
            w(f"- Con truncamiento: **{len(truncated)}**")
            w("")
            if truncated:
                w("Carreras con datos truncados (primeras 20):")
                w("")
                w("| URL | Devueltos | Total API | Truncado |")
                w("|---|---:|---:|:---:|")
                for c in sorted(truncated, key=lambda x: x["url"])[:20]:
                    w(f"| `{c['url']}` | {c['returned']} | {c['total']} | **SÍ** |")
                if len(truncated) > 20:
                    w(f"\n_(+{len(truncated) - 20} carreras más truncadas)_")
                w("")

        return "\n".join(L)

    def write_markdown(self, path: str, title: str = "Benchmark build_f1_csv.py") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(title=title), encoding="utf-8")