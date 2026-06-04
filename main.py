"""
main.py - Punto de entrada principal
Algoritmo Genético NSGA-II para Asignación de Camas UCI
Hospital Regional de Lambayeque

Uso:
    python main.py                    # Ejecutar con datos simulados
    python main.py --gens 50          # Reducir generaciones (prueba rápida)
    python main.py --doctors 4        # Turno con 4 médicos (máx 24 camas)
    python main.py --no-plots         # Sin gráficos
"""
from __future__ import annotations
import argparse
import os
import sys
import random
import time

# ── Asegurar que el directorio del proyecto esté en sys.path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TAMAÑO_POBLACIÓN, GENERACIONES_MAX,
    OPCIONES_MÉDICOS_TURNO, SEMILLA_ALEATORIA,
    HORIZONTE_PLANIFICACIÓN,
)
import config as cfg
from patients import generar_pacientes_actuales, generar_lista_espera_electivos
from engine import NSGA2_UCI


# ══════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      NSGA-II · Optimización UCI · Hospital Lambayeque        ║
║      28 camas  |  Adulto / Pediátrico / Neonatal             ║
╚══════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════
def analizar_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSGA-II UCI - Hospital Lambayeque")
    p.add_argument("--gens",     type=int, default=GENERACIONES_MAX,
                   help=f"Generaciones máximas (default: {GENERACIONES_MAX})")
    p.add_argument("--pop",      type=int, default=TAMAÑO_POBLACIÓN,
                   help=f"Tamaño de población (default: {TAMAÑO_POBLACIÓN})")
    p.add_argument("--doctors",  type=int, choices=[3, 4, 5], default=5,
                   help="Médicos por turno: 3→18 camas, 4→24, 5→28 (default: 5)")
    p.add_argument("--seed",     type=int, default=SEMILLA_ALEATORIA,
                   help=f"Semilla aleatoria (default: {SEMILLA_ALEATORIA})")
    p.add_argument("--no-plots", action="store_true",
                   help="No generar gráficos")
    p.add_argument("--out-dir",  type=str, default="resultados",
                   help="Directorio de salida (default: resultados)")
    return p.parse_args()


# ══════════════════════════════════════════════
#  PRINCIPAL
# ══════════════════════════════════════════════
def principal() -> None:
    args = analizar_argumentos()
    print(BANNER)

    # ── Aplicar configuración dinámica ──
    cfg.GENERACIONES_MAX  = args.gens
    cfg.TAMAÑO_POBLACIÓN  = args.pop
    cfg.CAMAS_ACTIVAS_MAX  = OPCIONES_MÉDICOS_TURNO[args.doctors]
    cfg.MÉDICOS_POR_DEFECTO  = args.doctors

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"  Médicos por turno : {args.doctors}  →  máx {cfg.CAMAS_ACTIVAS_MAX} camas activas")
    print(f"  Generaciones      : {args.gens}")
    print(f"  Población         : {args.pop}")
    print(f"  Semilla aleatoria : {args.seed}")
    print()

    # ── Generar datos hospitalarios ──
    rng = random.Random(args.seed)
    pacientes_actuales  = generar_pacientes_actuales(rng)
    lista_espera_electivos = generar_lista_espera_electivos(rng, n=40)

    print(f"  Pacientes actuales en UCI : {len(pacientes_actuales)}")
    print(f"  Pacientes en lista espera : {len(lista_espera_electivos)}")
    print()

    # ── Mostrar distribución actual de pacientes ──
    from config import CAMAS
    for tipo_paciente in ["adulto", "pediátrico", "neonatal"]:
        n  = sum(1 for p in pacientes_actuales if p.tipo_paciente == tipo_paciente)
        cap = len(CAMAS[tipo_paciente])
        print(f"  UCI {tipo_paciente.upper():<12}: {n}/{cap} camas ocupadas")
    print()

    # ── Ejecutar NSGA-II ──
    t_start = time.time()
    motor = NSGA2_UCI(
        pacientes_actuales  = pacientes_actuales,
        lista_espera_electivos = lista_espera_electivos,
        semilla              = args.seed,
    )
    frente_pareto = motor.ejecutar(verboso=True)
    elapsed = time.time() - t_start

    print(f"\n  Tiempo total: {elapsed:.1f}s")

    # ── Mostrar reportes en consola integrados de visualize.py ──
    _imprimir_resumen_pareto_consola(frente_pareto)
    
    # Buscar el mejor individuo con mayor ocupación (menor valor en aptitud[0] debido a la minimización)
    if frente_pareto:
        mejor_individuo = min(frente_pareto, key=lambda ind: ind.aptitud[0])
        _imprimir_programa_consola(mejor_individuo, "Mejor Agenda (mayor ocupación)")

    # ── Exportar resultados CSV ──
    _exportar_csv(frente_pareto, motor.historial, args.out_dir)

    print("\n  ✓ Proceso completado.\n")


# ══════════════════════════════════════════════
#  MÉTODOS REUBICADOS DE VISUALIZE.PY
# ══════════════════════════════════════════════
def _imprimir_resumen_pareto_consola(frente_pareto: list) -> None:
    print("\n" + "═" * 65)
    print("  FRENTE DE PARETO ÓPTIMO - UCI Hospital Regional Lambayeque")
    print("═" * 65)
    print(f"  {'#':>3}  {'Ocupación':>10}  {'Retraso':>8}  {'Emergencias':>12}  {'Rango':>5}")
    print("─" * 65)
    for i, individuo in enumerate(frente_pareto, 1):
        occ   = (1.0 - individuo.aptitud[0]) * 100
        retraso = individuo.aptitud[1]
        emerg = (1.0 - individuo.aptitud[2]) * 100
        print(f"  {i:>3}  {occ:>9.1f}%  {retraso:>8.2f}  {emerg:>11.1f}%  {individuo.rango:>5}")
    print("═" * 65)


def _imprimir_programa_consola(individuo: any, title: str = "Agenda Semanal UCI") -> None:
    iconos_categorías = {"actual": "🟢", "emergencia": "🔵", "electivo": "🟠"}

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

    for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
        # Protegemos el acceso en caso de que el programa guarde las claves de forma diferente
        electivos = individuo.programa.get(día, []) if hasattr(individuo.programa, 'get') else individuo.programa[día]
        
        # Filtrar solo pacientes de tipo 'electivo' tal como requería tu output original
        electivos_filtrados = [p for p in electivos if getattr(p, 'categoría', '') == 'electivo']
        
        print(f"\nDÍA {día} ({len(electivos_filtrados)} electivos programados)")
        if not electivos_filtrados:
            print("    (sin pacientes electivos)")
            continue
            
        for p in electivos:
            ico = iconos_categorías.get(p.categoría, "⚪")
            retraso_str = f" [retraso: {p.retraso}d]" if p.retraso > 0 else ""
            print(
                f"    {ico} ID:{p.id_paciente:<5} "
                f"Tipo:{p.tipo_paciente:<11} "
                f"LOS:{p.los}d  "
                f"LoC:{p.pérdida_oportunidad}"
                f"{retraso_str}"
            )


# ══════════════════════════════════════════════
#  EXPORTAR CSV
# ══════════════════════════════════════════════
def _exportar_csv(
    frente_pareto:  list,
    historial: list[dict],
    out_dir: str,
) -> None:
    import csv

    # Frente de Pareto
    ruta_pareto = os.path.join(out_dir, "soluciones_pareto.csv")
    with open(ruta_pareto, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Ocupacion_%", "Indice_Retraso", "Tasa_Emergencias_%", "Rango"])
        for i, individuo in enumerate(frente_pareto, 1):
            writer.writerow([
                i,
                round((1 - individuo.aptitud[0]) * 100, 2),
                round(individuo.aptitud[1], 4),
                round((1 - individuo.aptitud[2]) * 100, 2),
                individuo.rango,
            ])
    print(f"  CSV Pareto guardado: {ruta_pareto}")

    # Convergencia
    ruta_convergencia = os.path.join(out_dir, "convergencia.csv")
    if historial:
        with open(ruta_convergencia, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Generacion", "Tamaño_Pareto", "Mejor_Ocupacion_%",
                             "Ocupacion_Promedio_%", "Mejor_Retraso", "Mejor_Emergencia_%"])
            for h in historial:
                writer.writerow([
                    h.get("generación"),
                    h.get("tamaño_pareto"),
                    round(h.get("mejor_ocupación", 0) * 100, 2),
                    round(h.get("ocupación_promedio",  0) * 100, 2),
                    round(h.get("mejor_retraso", 0), 4),
                    round(h.get("mejor_emergencia", 0) * 100, 2),
                ])
        print(f"  CSV Convergencia guardado: {ruta_convergencia}")


if __name__ == "__main__":
    principal()