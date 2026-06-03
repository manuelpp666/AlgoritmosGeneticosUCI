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
    POPULATION_SIZE, MAX_GENERATIONS,
    DOCTORS_PER_SHIFT_OPTIONS, RANDOM_SEED,
)
import config as cfg
from patients import generate_current_patients, generate_elective_waitlist
from engine import NSGAII_UCI
from visualize import (
    print_pareto_summary,
    print_schedule,
    plot_pareto_front,
    plot_convergence,
    plot_bed_occupancy_heatmap,
)


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
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSGA-II UCI - Hospital Lambayeque")
    p.add_argument("--gens",     type=int, default=MAX_GENERATIONS,
                   help=f"Generaciones máximas (default: {MAX_GENERATIONS})")
    p.add_argument("--pop",      type=int, default=POPULATION_SIZE,
                   help=f"Tamaño de población (default: {POPULATION_SIZE})")
    p.add_argument("--doctors",  type=int, choices=[3, 4, 5], default=5,
                   help="Médicos por turno: 3→18 camas, 4→24, 5→28 (default: 5)")
    p.add_argument("--seed",     type=int, default=RANDOM_SEED,
                   help=f"Semilla aleatoria (default: {RANDOM_SEED})")
    p.add_argument("--no-plots", action="store_true",
                   help="No generar gráficos")
    p.add_argument("--out-dir",  type=str, default="results",
                   help="Directorio de salida (default: results)")
    return p.parse_args()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    print(BANNER)

    # ── Aplicar configuración dinámica ──
    cfg.MAX_GENERATIONS  = args.gens
    cfg.POPULATION_SIZE  = args.pop
    cfg.MAX_BEDS_ACTIVE  = DOCTORS_PER_SHIFT_OPTIONS[args.doctors]
    cfg.DEFAULT_DOCTORS  = args.doctors

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"  Médicos por turno : {args.doctors}  →  máx {cfg.MAX_BEDS_ACTIVE} camas activas")
    print(f"  Generaciones      : {args.gens}")
    print(f"  Población         : {args.pop}")
    print(f"  Semilla aleatoria : {args.seed}")
    print()

    # ── Generar datos hospitalarios ──
    rng = random.Random(args.seed)
    current_patients  = generate_current_patients(rng)
    elective_waitlist = generate_elective_waitlist(rng, n=40)

    print(f"  Pacientes actuales en UCI : {len(current_patients)}")
    print(f"  Pacientes en lista espera : {len(elective_waitlist)}")
    print()

    # ── Mostrar distribución actual de pacientes ──
    from config import BEDS
    for ptype in ["adult", "pediatric", "neonatal"]:
        n  = sum(1 for p in current_patients if p.patient_type == ptype)
        cap = len(BEDS[ptype])
        print(f"  UCI {ptype.upper():<12}: {n}/{cap} camas ocupadas")
    print()

    # ── Ejecutar NSGA-II ──
    t_start = time.time()
    engine = NSGAII_UCI(
        current_patients  = current_patients,
        elective_waitlist = elective_waitlist,
        seed              = args.seed,
    )
    pareto = engine.run(verbose=True)
    elapsed = time.time() - t_start

    print(f"\n  Tiempo total: {elapsed:.1f}s")

    # ── Reporte en consola ──
    print_pareto_summary(pareto)

    # ── Mejor solución por ocupación ──
    if pareto:
        best_occ = min(pareto, key=lambda x: x.fitness[0])   # mínimo = máx ocupación
        print_schedule(best_occ, title="Mejor Agenda (mayor ocupación)")

    # ── Gráficos ──
    if not args.no_plots:
        plot_pareto_front(pareto, out_dir=args.out_dir)
        plot_convergence(engine.history, out_dir=args.out_dir)
        if pareto:
            best_occ = min(pareto, key=lambda x: x.fitness[0])
            plot_bed_occupancy_heatmap(best_occ, out_dir=args.out_dir)
        print(f"\n  Gráficos guardados en: ./{args.out_dir}/")

    # ── Exportar resultados CSV ──
    _export_csv(pareto, engine.history, args.out_dir)

    print("\n  ✓ Proceso completado.\n")


# ══════════════════════════════════════════════
#  EXPORTAR CSV
# ══════════════════════════════════════════════
def _export_csv(
    pareto:  list,
    history: list[dict],
    out_dir: str,
) -> None:
    import csv

    # Frente de Pareto
    pareto_path = os.path.join(out_dir, "pareto_solutions.csv")
    with open(pareto_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Ocupacion_%", "Indice_Retraso", "Tasa_Emergencias_%", "Rank"])
        for i, ind in enumerate(pareto, 1):
            writer.writerow([
                i,
                round((1 - ind.fitness[0]) * 100, 2),
                round(ind.fitness[1], 4),
                round((1 - ind.fitness[2]) * 100, 2),
                ind.rank,
            ])
    print(f"  Pareto CSV guardado: {pareto_path}")

    # Convergencia
    conv_path = os.path.join(out_dir, "convergence.csv")
    if history:
        with open(conv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Generacion", "Pareto_Size", "Best_Occ_%",
                             "Avg_Occ_%", "Best_Delay", "Best_Emerg_%"])
            for h in history:
                writer.writerow([
                    h.get("generation"),
                    h.get("pareto_size"),
                    round(h.get("best_occ", 0) * 100, 2),
                    round(h.get("avg_occ",  0) * 100, 2),
                    round(h.get("best_delay", 0), 4),
                    round(h.get("best_emerg", 0) * 100, 2),
                ])
        print(f"  Convergencia CSV guardada: {conv_path}")


if __name__ == "__main__":
    main()
