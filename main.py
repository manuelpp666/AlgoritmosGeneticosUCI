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
from chromosome import Individuo


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
#  IMPRESIÓN DE AGENDAS DETALLADAS
# ══════════════════════════════════════════════
def imprimir_agenda_detallada(
    individuo: Individuo,
    nombre_estrategia: str,
    ocupacion: float,
    retraso: float,
    tasa_emerg: float,
) -> None:
    """
    Imprime la agenda semanal detallada de un individuo con formato bonito.
    """
    print(f"\n────────────────────────────────────────────────────────")
    print(f"  {nombre_estrategia}")
    print(f"────────────────────────────────────────────────────────")
    print(f"  Ocupación: {ocupacion:.1%}  |  Retraso: {retraso:.2f}  |  Emergencias: {tasa_emerg:.1%}")
    print()

    for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
        pacientes_día = individuo.programa.get(día, [])
        n_electivos = len(pacientes_día)

        print(f"  DÍA {día}  ({n_electivos} electivos programados)")
        if not pacientes_día:
            print(f"    (sin pacientes electivos)")
        else:
            for p in pacientes_día:
                retraso_txt = f"  [retraso: {p.retraso}d]" if p.retraso > 0 else ""
                print(
                    f"    🟠 ID:{p.id_paciente:<6} Tipo:{p.tipo_paciente:<12} "
                    f"LOS:{p.los}d  LoC:{p.pérdida_oportunidad}{retraso_txt}"
                )
        print()


def imprimir_3_estrategias_extremas(
    motor: "NSGA2_UCI",
    lista_espera_electivos: list,
) -> None:
    """
    Imprime las 3 mejores estrategias (extremos del Frente de Pareto).
    """
    est1, est2, est3 = motor.obtener_3_estrategias_extremas()
    estrategias = [e for e in [est1, est2, est3] if e is not None]

    if not estrategias:
        print("\n  ⚠ No hay soluciones en el Frente de Pareto.")
        return

    titulos = [
        "Mejor Agenda (Mayor Ocupación)",
        "Estrategia 2 (Equilibrada)" if len(estrategias) > 1 else "",
        "Estrategia 3 (Menor Retraso)" if len(estrategias) > 2 else "",
    ]

    print("\n═════════════════════════════════════════════════════════════════")
    print("  FRENTE DE PARETO ÓPTIMO - UCI Hospital Regional Lambayeque")
    print("═════════════════════════════════════════════════════════════════")

    # Resumen en tabla
    print("    #   Ocupación   Retraso   Emergencias  Rango")
    print("─────────────────────────────────────────────────────────────────")
    for i, est in enumerate(estrategias, 1):
        ocup = (1.0 - est.aptitud[0]) * 100
        ret = est.aptitud[1]
        emerg = (1.0 - est.aptitud[2]) * 100
        print(f"    {i}       {ocup:>5.1f}%      {ret:>4.2f}         {emerg:>5.1f}%      {est.rango}")

    print("═════════════════════════════════════════════════════════════════")

    # Agendas detalladas
    for i, est in enumerate(estrategias, 1):
        ocup_pct = (1.0 - est.aptitud[0]) * 100
        ret = est.aptitud[1]
        emerg_pct = (1.0 - est.aptitud[2]) * 100
        imprimir_agenda_detallada(
            est,
            f"Estrategia {i}",
            ocup_pct / 100.0,
            ret,
            emerg_pct / 100.0,
        )


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

    # ── Mostrar 3 estrategias extremas ──
    imprimir_3_estrategias_extremas(motor, lista_espera_electivos)

    # ── Exportar resultados CSV ──
    _exportar_csv(frente_pareto, motor.historial, args.out_dir)
    _exportar_agendas_csv(motor, args.out_dir)

    print("\n  ✓ Proceso completado.\n")

# ══════════════════════════════════════════════
#  MÉTODOS REUBICADOS DE VISUALIZE.PY
# ══════════════════════════════════════════════


# ══════════════════════════════════════════════
#  EXPORTAR CSV
def _exportar_csv(
    frente_pareto:  list,
    historial: list[dict],
    out_dir: str,
) -> None:
    import csv

    # Frente de Pareto - Métricas
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


def _exportar_agendas_csv(
    motor: "NSGA2_UCI",
    out_dir: str,
) -> None:
    """
    Exporta las agendas de las 3 estrategias extremas a archivos CSV detallados.
    """
    import csv
    
    est1, est2, est3 = motor.obtener_3_estrategias_extremas()
    estrategias = [
        (est1, "estrategia_1_max_ocupacion"),
        (est2, "estrategia_2_equilibrada"),
        (est3, "estrategia_3_otros"),
    ]

    for est, nombre_archivo in estrategias:
        if est is None:
            continue

        ruta_agenda = os.path.join(out_dir, f"agenda_{nombre_archivo}.csv")
        with open(ruta_agenda, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Encabezado con métricas
            ocup_pct = (1.0 - est.aptitud[0]) * 100
            ret = est.aptitud[1]
            emerg_pct = (1.0 - est.aptitud[2]) * 100
            writer.writerow(["MÉTRICAS"])
            writer.writerow(["Ocupación", "Retraso", "Emergencias"])
            writer.writerow([f"{ocup_pct:.1f}%", f"{ret:.2f}", f"{emerg_pct:.1f}%"])
            writer.writerow([])  # Línea en blanco
            
            # Agendas por día
            writer.writerow(["DÍA", "ID_Paciente", "Tipo", "LOS", "LoC", "Retraso"])
            for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
                pacientes_día = est.programa.get(día, [])
                for p in pacientes_día:
                    writer.writerow([
                        día,
                        p.id_paciente,
                        p.tipo_paciente,
                        p.los,
                        p.pérdida_oportunidad,
                        p.retraso,
                    ])
        
        print(f"  CSV Agenda guardado: {ruta_agenda}")


if __name__ == "__main__":
    principal()