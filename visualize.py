"""
visualize.py - Visualización de resultados y reportes
Hospital Regional de Lambayeque - NSGA-II UCI
"""
from __future__ import annotations
import os
from typing import Optional

# Importaciones opcionales (graficar solo si matplotlib está instalado)
try:
    import matplotlib
    matplotlib.use("Agg")          # no requiere pantalla
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from chromosome import Individual
from config import PLANNING_HORIZON, MAX_BEDS_ACTIVE, BEDS


# ══════════════════════════════════════════════
#  REPORTE EN CONSOLA
# ══════════════════════════════════════════════
def print_pareto_summary(pareto: list[Individual]) -> None:
    print("\n" + "═" * 65)
    print("  FRENTE DE PARETO ÓPTIMO - UCI Hospital Regional Lambayeque")
    print("═" * 65)
    print(f"  {'#':>3}  {'Ocupación':>10}  {'Delay':>8}  {'Emergencias':>12}  {'Rank':>5}")
    print("─" * 65)
    for i, ind in enumerate(pareto, 1):
        occ   = (1.0 - ind.fitness[0]) * 100
        delay = ind.fitness[1]
        emerg = (1.0 - ind.fitness[2]) * 100
        print(f"  {i:>3}  {occ:>9.1f}%  {delay:>8.2f}  {emerg:>11.1f}%  {ind.rank:>5}")
    print("═" * 65)


def print_schedule(individual: Individual, title: str = "Agenda Semanal UCI") -> None:
    """Imprime en consola el schedule de un individuo."""
    cat_colors = {"current": "🟢", "emergency": "🔵", "elective": "🟠"}
    bed_types = {**{f"Adult {i+1}": "adult" for i in range(18)}}

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

    for day in range(1, PLANNING_HORIZON + 1):
        electives = individual.schedule[day]
        print(f"\n  DÍA {day}  ({len(electives)} electivos programados)")
        if not electives:
            print("    (sin pacientes electivos)")
        for p in electives:
            ico = cat_colors.get(p.category, "⚪")
            delay_str = f"  [retraso: {p.delay}d]" if p.delay > 0 else ""
            print(
                f"    {ico} ID:{p.patient_id:<8} "
                f"Tipo:{p.patient_type:<12} "
                f"LOS:{p.los}d  "
                f"LoC:{p.loss_of_chance}"
                f"{delay_str}"
            )


# ══════════════════════════════════════════════
#  GRÁFICOS (requiere matplotlib)
# ══════════════════════════════════════════════
def plot_pareto_front(
    pareto: list[Individual],
    out_dir: str = ".",
) -> Optional[str]:
    """
    Grafica el frente de Pareto 3D (f1 vs f2 vs f3).
    Guarda el gráfico como PNG y retorna la ruta.
    """
    if not HAS_MPL:
        print("[Visualize] matplotlib no disponible. Instala con: pip install matplotlib")
        return None

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    occs   = [(1 - ind.fitness[0]) * 100 for ind in pareto]
    delays = [ind.fitness[1]              for ind in pareto]
    emergs = [(1 - ind.fitness[2]) * 100  for ind in pareto]

    sc = ax.scatter(occs, delays, emergs, c=emergs, cmap="plasma", s=60, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Tasa Admisión Emergencias (%)")

    ax.set_xlabel("Tasa Ocupación (%)", labelpad=10)
    ax.set_ylabel("Índice Retraso", labelpad=10)
    ax.set_zlabel("Tasa Emergencias (%)", labelpad=10)
    ax.set_title("Frente de Pareto - UCI NSGA-II\nHospital Regional Lambayeque", pad=15)

    path = os.path.join(out_dir, "pareto_front_3d.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Gráfico Pareto guardado: {path}")
    return path


def plot_convergence(history: list[dict], out_dir: str = ".") -> Optional[str]:
    """Grafica la convergencia de las métricas a lo largo de las generaciones."""
    if not HAS_MPL or not history:
        return None

    gens   = [h["generation"]  for h in history]
    occs   = [h.get("best_occ", 0) * 100 for h in history]
    delays = [h.get("best_delay", 0) for h in history]
    emergs = [h.get("best_emerg", 0) * 100 for h in history]
    pareto = [h.get("pareto_size", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Convergencia NSGA-II - UCI Hospital Lambayeque", fontsize=13)

    axes[0, 0].plot(gens, occs, color="#2196F3")
    axes[0, 0].set_title("Mejor Tasa Ocupación (%)")
    axes[0, 0].set_xlabel("Generación")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(gens, delays, color="#F44336")
    axes[0, 1].set_title("Mejor Índice Retraso (mín)")
    axes[0, 1].set_xlabel("Generación")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(gens, emergs, color="#4CAF50")
    axes[1, 0].set_title("Mejor Tasa Admisión Emergencias (%)")
    axes[1, 0].set_xlabel("Generación")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(gens, pareto, color="#FF9800")
    axes[1, 1].set_title("Tamaño Frente de Pareto")
    axes[1, 1].set_xlabel("Generación")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Gráfico convergencia guardado: {path}")
    return path


def plot_bed_occupancy_heatmap(
    individual: Individual,
    out_dir: str = ".",
    title: str = "Ocupación de Camas UCI",
) -> Optional[str]:
    """
    Genera un heatmap de ocupación de camas por día (electivos).
    Filas = camas, Columnas = días.
    Colores: verde=actual, azul=emergencia, naranja=electivo, gris=vacía
    """
    if not HAS_MPL:
        return None

    import numpy as np

    # Construir matriz de ocupación [cama, día]
    n_beds = 28
    grid   = [["" for _ in range(PLANNING_HORIZON)] for _ in range(n_beds)]

    for day, plist in individual.schedule.items():
        for p in plist:
            if p.assigned_bed is not None:
                bid = p.assigned_bed - 1  # 0-indexed
                if 0 <= bid < n_beds:
                    grid[bid][day - 1] = p.category

    # Mapeo de categoría a valor numérico
    cat_map = {"current": 1, "emergency": 2, "elective": 3, "": 0}
    data = [[cat_map.get(cell, 0) for cell in row] for row in grid]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.colors.ListedColormap(["#ECEFF1", "#81C784", "#64B5F6", "#FFB74D"])
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Etiquetas de ejes
    ax.set_xticks(range(PLANNING_HORIZON))
    ax.set_xticklabels([f"Día {d+1}" for d in range(PLANNING_HORIZON)])
    ax.set_yticks(range(n_beds))

    # Etiquetas de camas con tipo
    bed_labels = []
    for bid in range(1, n_beds + 1):
        from config import BED_TYPE_MAP
        btype = BED_TYPE_MAP.get(bid, "?")
        type_abbr = {"adult": "A", "pediatric": "P", "neonatal": "N"}.get(btype, "?")
        bed_labels.append(f"Cama {bid} ({type_abbr})")
    ax.set_yticklabels(bed_labels, fontsize=7)

    # Leyenda
    legend_elements = [
        mpatches.Patch(color="#ECEFF1", label="Vacía"),
        mpatches.Patch(color="#81C784", label="Paciente Actual"),
        mpatches.Patch(color="#64B5F6", label="Emergencia"),
        mpatches.Patch(color="#FFB74D", label="Electivo"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Separadores por tipo de cama
    ax.axhline(y=17.5, color="black", linewidth=1.5, linestyle="--")
    ax.axhline(y=21.5, color="black", linewidth=1.5, linestyle="--")
    ax.text(PLANNING_HORIZON + 0.1, 9,   "ADULTO",    fontsize=7, rotation=90, va="center")
    ax.text(PLANNING_HORIZON + 0.1, 19.5,"PED.",      fontsize=7, rotation=90, va="center")
    ax.text(PLANNING_HORIZON + 0.1, 25,  "NEONATAL",  fontsize=7, rotation=90, va="center")

    ax.set_title(f"{title}\n(Solución Pareto - mejor ocupación)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "bed_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Heatmap de camas guardado: {path}")
    return path
