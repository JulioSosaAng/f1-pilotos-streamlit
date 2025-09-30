# build_f1_csv.py
# -*- coding: utf-8 -*-
"""
Rellena un CSV con datos de F1 (2024 completa y 2025 hasta Monza inclusive)
usando la API pública Ergast. Intenta fallback a la API sucesora (jolpica-f1)
si Ergast no responde.

Campos que se completan automáticamente (si están disponibles en la API):
- season, round, gp_name, circuit, country, date
- sprint_weekend (sí/no)
- pole_driver, pole_team, pole_time
- winner_driver, winner_team, p2_driver, p3_driver
- fastest_lap_driver, fastest_lap_team, fastest_lap_time, fastest_lap_lap
- dnfs (conteo por status no "Finished"/"+n Laps")
- avg_pitstops_per_driver (desde 2012, usando endpoint /pitstops)

Campos que quedan en blanco por no estar en Ergast: laps_scheduled, circuit_length_km,
race_distance_km, safety_cars, vscs, red_flags, weather_*, attendance (puedes
rellenarlos manualmente o integrar otras fuentes).

Uso:
  pip install pandas requests
  python build_f1_csv.py --out f1_2024_2025_filled.csv

Opcional (si ya tienes un template CSV con columnas): puedes pasar --template
y el script intentará hacer merge por (season, round).
"""
from __future__ import annotations
import argparse
import sys
import time
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd

ERGAST_BASE = "https://ergast.com/api/f1"
# Fallback (sucesor compatible): https://github.com/jolpica/jolpica-f1
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

HEADERS = {"User-Agent": "F1 Data Builder (github.com/your-handle)"}

def _get_json(url: str) -> Optional[Dict[str, Any]]:
    for base in (None,):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return None

def ergast_json(path: str) -> Optional[Dict[str, Any]]:
    # Intenta Ergast, luego Jolpica (mismo path)
    for base in (ERGAST_BASE, JOLPICA_BASE):
        url = f"{base}{path}.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None

def season_schedule(season: int) -> List[Dict[str, Any]]:
    data = ergast_json(f"/{season}")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", []) if data else []
    return races

def get_rounds_up_to_monza(season: int) -> List[int]:
    races = season_schedule(season)
    if not races:
        return []
    rounds = []
    monza_round = None
    for r in races:
        try:
            rnd = int(r.get("round"))
        except Exception:
            continue
        rounds.append(rnd)
        # Detectar Monza por nombre de circuito
        circuit_name = (r.get("Circuit", {}) or {}).get("circuitName", "").lower()
        if "monza" in circuit_name:
            monza_round = rnd
    if season == 2025 and monza_round is not None:
        return [rnd for rnd in rounds if rnd <= monza_round]
    return rounds

def fetch_qualifying(season: int, rnd: int) -> Tuple[str, str, str]:
    """Devuelve (pole_driver, pole_team, pole_time)"""
    data = ergast_json(f"/{season}/{rnd}/qualifying")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", []) if data else []
    if not races:
        return "", "", ""
    qres = races[0].get("QualifyingResults", [])
    if not qres:
        return "", "", ""
    pole = qres[0]  # posición 1
    drv = pole.get("Driver", {})
    cons = pole.get("Constructor", {})
    pole_driver = f"{drv.get('givenName','')} {drv.get('familyName','')}".strip()
    pole_team = cons.get("name", "")
    # Preferir Q3, luego Q2, Q1
    pole_time = pole.get("Q3") or pole.get("Q2") or pole.get("Q1") or ""
    return pole_driver, pole_team, pole_time

def fetch_race_results(season: int, rnd: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = ergast_json(f"/{season}/{rnd}/results")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", []) if data else []
    if not races:
        return {}, []
    return races[0], races[0].get("Results", [])

def extract_podium_and_fastest(results: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    out = {"winner_driver":"","winner_team":"","p2_driver":"","p3_driver":""}
    # Podio
    for res in results:
        pos = res.get("position")
        drv = res.get("Driver", {}); cons = res.get("Constructor", {})
        name = f"{drv.get('givenName','')} {drv.get('familyName','')}".strip()
        team = cons.get("name","")
        if pos == "1":
            out["winner_driver"] = name; out["winner_team"] = team
        elif pos == "2":
            out["p2_driver"] = name
        elif pos == "3":
            out["p3_driver"] = name
    # Vuelta rápida
    fl_entry = None
    for res in results:
        fl = res.get("FastestLap")
        if fl and str(fl.get("rank")) == "1":
            fl_entry = {"driver": res.get("Driver", {}), "constructor": res.get("Constructor", {}), "fl": fl}
            break
    return out, (fl_entry or {})

def compute_dnfs(results: List[Dict[str, Any]]) -> int:
    dnfs = 0
    for res in results:
        status = (res.get("status") or "").lower()
        # finished variations include "+1 lap", "+2 laps", "finished"
        if ("finished" in status) or ("lap" in status and "+" in status):
            continue
        dnfs += 1
    return dnfs

def compute_avg_pitstops(season: int, rnd: int) -> str:
    # Disponible desde 2012
    data = ergast_json(f"/{season}/{rnd}/pitstops?limit=2000")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", []) if data else []
    if not races:
        return ""
    stops = races[0].get("PitStops", [])
    if not stops:
        return "0"
    # contar por driverId
    by_driver: Dict[str,int] = {}
    for s in stops:
        did = s.get("driverId","")
        by_driver[did] = by_driver.get(did, 0) + 1
    if not by_driver:
        return ""
    avg = sum(by_driver.values()) / len(by_driver)
    return f"{avg:.2f}"

def has_sprint(season: int, rnd: int) -> str:
    data = ergast_json(f"/{season}/{rnd}/sprint")
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", []) if data else []
    return "sí" if races else "no"

def fill_row_from_api(season: int, rnd: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    # schedule/base info
    sched = season_schedule(season)
    race_info = next((r for r in sched if int(r.get("round", -1)) == rnd), None)
    if not race_info:
        return {}
    row["season"] = season
    row["round"] = rnd
    row["gp_name"] = race_info.get("raceName","")
    row["circuit"] = (race_info.get("Circuit",{}) or {}).get("circuitName","")
    row["country"] = ((race_info.get("Circuit",{}) or {}).get("Location",{}) or {}).get("country","")
    row["date"] = race_info.get("date","")

    row["sprint_weekend"] = has_sprint(season, rnd)

    # qualifying / pole
    pole_driver, pole_team, pole_time = fetch_qualifying(season, rnd)
    row["pole_driver"] = pole_driver
    row["pole_team"] = pole_team
    row["pole_time"] = pole_time

    # race results / podium / fastest
    race_hdr, results = fetch_race_results(season, rnd)
    pod, fl = extract_podium_and_fastest(results)
    row.update(pod)

    if fl:
        drv = fl["driver"]; cons = fl["constructor"]; flv = fl["fl"]
        row["fastest_lap_driver"] = f"{drv.get('givenName','')} {drv.get('familyName','')}".strip()
        row["fastest_lap_team"] = cons.get("name","")
        row["fastest_lap_time"] = ((flv.get("Time") or {}).get("time") or "")
        row["fastest_lap_lap"] = flv.get("lap","")
    else:
        row["fastest_lap_driver"] = ""
        row["fastest_lap_team"] = ""
        row["fastest_lap_time"] = ""
        row["fastest_lap_lap"] = ""

    row["dnfs"] = compute_dnfs(results) if results else ""
    row["avg_pitstops_per_driver"] = compute_avg_pitstops(season, rnd)

    # placeholders (no disponibles en Ergast)
    for k in ["laps_scheduled","circuit_length_km","race_distance_km","safety_cars","vscs","red_flags",
              "weather_summary","air_temp_c","track_temp_c","rain_probability_pct","attendance","notes"]:
        row.setdefault(k, "")
    return row

def build_dataframe() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # 2024: todas las rondas disponibles en calendario
    for rnd in get_rounds_up_to_monza(2024):  # en 2024 devolvemos todas
        rows.append(fill_row_from_api(2024, rnd))

    # 2025: solo hasta Monza inclusive
    for rnd in get_rounds_up_to_monza(2025):
        rows.append(fill_row_from_api(2025, rnd))

    # limpiar vacíos
    rows = [r for r in rows if r]
    cols = [
        "season","round","gp_name","circuit","country","date",
        "laps_scheduled","circuit_length_km","race_distance_km","sprint_weekend",
        "pole_driver","pole_team","pole_time",
        "winner_driver","winner_team","p2_driver","p3_driver",
        "fastest_lap_driver","fastest_lap_team","fastest_lap_time","fastest_lap_lap",
        "safety_cars","vscs","red_flags","dnfs","avg_pitstops_per_driver",
        "weather_summary","air_temp_c","track_temp_c","rain_probability_pct",
        "attendance","notes"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["season","round"]).reset_index(drop=True)
    return df

def merge_with_template(df_new: pd.DataFrame, template_path: Optional[str]) -> pd.DataFrame:
    if not template_path:
        return df_new
    try:
        tpl = pd.read_csv(template_path)
    except Exception as e:
        print(f"[warn] No pude leer template: {e}")
        return df_new
    # merge por season+round
    key = ["season","round"]
    tpl_cols = [c for c in tpl.columns if c not in key]
    merged = pd.merge(tpl, df_new, on=key, how="outer", suffixes=("_tpl",""))
    # preferir valores del df_new (API) si existen; mantener tpl si df_new vacío
    for col in [c for c in df_new.columns if c not in key]:
        col_tpl = f"{col}_tpl"
        if col_tpl in merged.columns:
            merged[col] = merged[col].where(merged[col].notna() & (merged[col]!=""), merged[col_tpl])
            merged.drop(columns=[col_tpl], inplace=True)
    # reordenar
    merged = merged[df_new.columns.tolist()]
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="f1_2024_2025_filled.csv")
    ap.add_argument("--template", default=None, help="Ruta a template CSV opcional para merge")
    args = ap.parse_args()

    print("Descargando y construyendo dataframe...")
    df = build_dataframe()
    if df.empty:
        print("No se obtuvieron datos. ¿API caída? Probá luego o cambiá de red.")
        sys.exit(1)

    if args.template:
        print(f"Haciendo merge con template: {args.template}")
        df = merge_with_template(df, args.template)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Listo: {args.out} con {len(df)} filas.")

if __name__ == "__main__":
    main()
