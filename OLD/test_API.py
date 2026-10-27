import requests
import json

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "F1 Test Script"}

def probar_todas_las_rondas_2026():
    season = 2026
    print(f"=== Probando temporada {season} completa ===")
    
    # 1. Traer el calendario de 2026
    url_season = f"{JOLPICA_BASE}/{season}.json"
    r = requests.get(url_season, headers=HEADERS, timeout=10)
    
    if r.status_code != 200:
        print(f"Error al obtener el calendario. Status: {r.status_code}")
        return

    data = r.json()
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    
    print(f"Total de carreras programadas en 2026: {len(races)}\n")

    # 2. Recorrer cada carrera del año
    for r_info in races:
        rnd = r_info.get("round")
        gp_name = r_info.get("raceName")
        date = r_info.get("date")
        
        # Consultar resultados de esta ronda
        url_results = f"{JOLPICA_BASE}/{season}/{rnd}/results.json"
        res_r = requests.get(url_results, headers=HEADERS, timeout=10)
        
        tiene_resultados = False
        ganador = "Sin disputar / Sin datos"
        
        if res_r.status_code == 200:
            res_data = res_r.json()
            res_races = res_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if res_races and res_races[0].get("Results"):
                tiene_resultados = True
                drv = res_races[0]["Results"][0].get("Driver", {})
                ganador = f"{drv.get('givenName', '')} {drv.get('familyName', '')}"

        print(f"Ronda {rnd}: {gp_name} ({date}) -> {'[DISPUTADA] Ganador: ' + ganador if tiene_resultados else '[FUTURA / SIN DATOS]'}")

if __name__ == "__main__":
    probar_todas_las_rondas_2026()