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
import random
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
ERGAST_BASE = "https://ergast.com/api/f1"
CHECKPOINT_FILE = "checkpoint.jsonl"

HEADERS = {"User-Agent": "F1 Data Builder (github.com/your-handle)"}

def _build_url(base: str, path: str) -> str:
    """Construye la URL manejando path con query string (ej: pitstops?limit=2000)."""
    if "?" in path:
        path_part, query = path.split("?", 1)
        return f"{base}{path_part}.json?{query}"
    return f"{base}{path}.json"

def ergast_json(path: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    for base in (JOLPICA_BASE, ERGAST_BASE):
        url = _build_url(base, path)
        
        for intento in range(max_retries):
            try:
                # Ya NO hacemos sleep antes del request
                r = requests.get(url, headers=HEADERS, timeout=30)
            except requests.RequestException as e:
                print(f"   [Red] Error en {url}: {e}")
                time.sleep((2 ** intento) + random.uniform(0, 0.5))
                continue

            # 1. Éxito
            if r.status_code == 200:
                return r.json()
            
            # 2. No existe (404) -> No reintentar
            if r.status_code == 404:
                return None
            
            # 3. Rate Limit (429) -> Respetar el Retry-After
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                # Si la API nos dice cuánto esperar, le hacemos caso. Si no, backoff exponencial.
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** intento)
                print(f"   [429] Límite alcanzado. Esperando {wait_time}s...")
                time.sleep(wait_time + random.uniform(0, 0.5))
                continue
            
            # 4. Errores del servidor (500+) -> Backoff exponencial
            if 500 <= r.status_code < 600:
                wait_time = (2 ** intento) + random.uniform(0, 0.5)
                print(f"   [{r.status_code}] Error servidor. Esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            
            # 5. Otros errores (400, 401, 403) -> Abortar
            print(f"   [{r.status_code}] Error de cliente en {url}. Abortando.")
            return None
            
    return None

@lru_cache(maxsize=16)

def season_schedule(season: int) -> Tuple[Dict[str, Any], ...]:
    # Nota: devolvemos una tupla en lugar de lista para asegurar que sea hashable, 
    # aunque lru_cache solo necesita que los argumentos lo sean. Pero es más seguro así.
    data = ergast_json(f"/{season}")
    if data:
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races:
            return tuple(races)
    return tuple()

def get_latest_available_rounds(season: int) -> List[int]:
    races = season_schedule(season)
    if not races:
        return []
    rounds = []
    for r in races:
        try:
            rnd = int(r.get("round"))
            rounds.append(rnd)
        except Exception:
            continue
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

def fill_row_from_api(season: int, rnd: int, race_info: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    # schedule/base info (ahora viene por parámetro)
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
    seasons_to_fetch = [2024, 2025, 2026]

    # 1. Cargar progreso anterior (NUEVO)
    checkpoint_data = {}
    if os.path.exists(CHECKPOINT_FILE):
        print(f"📦 Leyendo checkpoint anterior: {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    # Guardamos en un dict usando season_round como clave
                    checkpoint_data[f"{row.get('season')}_{row.get('round')}"] = row

    for season in seasons_to_fetch:
        print(f"\n--- Procesando temporada {season} ---")
        races = season_schedule(season)
        if not races:
            print(f"[Advertencia] No se pudo obtener el calendario para {season}")
            continue
            
        print(f"Total de rondas encontradas en {season}: {len(races)}")
        
        for r_info in races:
            try:
                rnd = int(r_info.get("round"))
            except Exception:
                continue
                
            gp_name = r_info.get("raceName", "Desconocido")
            date_str = r_info.get("date", "")
            
            if season == 2026 and date_str > pd.Timestamp.today().strftime('%Y-%m-%d'):
                continue

            # 2. Verificar si ya lo descargamos (NUEVO)
            key = f"{season}_{rnd}"
            if key in checkpoint_data:
                print(f"  -> Ronda {rnd} ({gp_name}): Ya en checkpoint, saltando...")
                rows.append(checkpoint_data[key])
                continue

            max_intentos = 3
            row = None
            
            for intento in range(1, max_intentos + 1):
                if intento == 1:
                    print(f"  -> Descargando datos ronda {rnd}: {gp_name}...")
                else:
                    print(f"     -> Reintentando ronda {rnd} (Intento {intento}/{max_intentos})...")
                
                row = fill_row_from_api(season, rnd, r_info)
                
                if row and (row.get("winner_driver") or season < 2026):
                    break
                
                time.sleep(1.5 * intento)

            if row and (row.get("winner_driver") or season < 2026):
                rows.append(row)
                # 3. Guardar la fila en el disco inmediatamente (NUEVO)
                with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            else:
                print(f"     [Aviso] La ronda {rnd} no devolvió resultados tras {max_intentos} intentos.")

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
    if not df.empty:
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
    from metrics import Metrics      
    M = Metrics()                          
    M.patch()                              
    M.start()                              

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="f1_2024_2025_2026_filled.csv")
    ap.add_argument("--template", default=None, help="Ruta a template CSV opcional para merge")
    args = ap.parse_args()
    metrics_out = f"metrics/{Path(args.out).stem}.md"

    print("Descargando y construyendo dataframe...")
    with M.phase("build_dataframe"):
        df = build_dataframe()
        
    if df.empty:
        print("No se obtuvieron datos. ¿API caída? Probá luego o cambiá de red.")
        M.stop()
        M.set_dataframe_info(df)                           
        M.write_markdown(metrics_out)
        sys.exit(1)

    if args.template:
        print(f"Haciendo merge con template: {args.template}")
        with M.phase("merge_template"):
            df = merge_with_template(df, args.template)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Listo: {args.out} con {len(df)} filas.")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint eliminado (extracción 100% completada).")

    M.stop()                               
    M.set_dataframe_info(df)               
    M.write_markdown(metrics_out)  
    print(f"Métricas guardadas en: {metrics_out}")

if __name__ == "__main__":
    main()
