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

from chromosome import Individuo
from config import HORIZONTE_PLANIFICACIÓN, CAMAS_ACTIVAS_MAX, CAMAS


# ══════════════════════════════════════════════
#  REPORTE EN CONSOLA
# ══════════════════════════════════════════════
def imprimir_resumen_pareto(frente_pareto: list[Individuo]) -> None:
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


def imprimir_programa(individuo: Individuo, title: str = "Agenda Semanal UCI") -> None:
    """Imprime en consola el programa de un individuo."""
    iconos_categorías = {"actual": "🟢", "emergencia": "🔵", "electivo": "🟠"}

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

    for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
        electivos = individuo.programa[día]
        print(f"\n  DÍA {día}  ({len(electivos)} electivos programados)")
        if not electivos:
            print("    (sin pacientes electivos)")
        for p in electivos:
            ico = iconos_categorías.get(p.categoría, "⚪")
            retraso_str = f"  [retraso: {p.retraso}d]" if p.retraso > 0 else ""
            print(
                f"    {ico} ID:{p.id_paciente:<8} "
                f"Tipo:{p.tipo_paciente:<12} "
                f"LOS:{p.los}d  "
                f"LoC:{p.pérdida_oportunidad}"
                f"{retraso_str}"
            )


# ══════════════════════════════════════════════
#  GRÁFICOS (requiere matplotlib)
# ══════════════════════════════════════════════
def graficar_frente_pareto(
    frente_pareto: list[Individuo],
    out_dir: str = ".",
) -> Optional[str]:
    """
    Grafica el frente de Pareto 3D (f1 vs f2 vs f3).
    Guarda el gráfico como PNG y retorna la ruta.
    """
    if not HAS_MPL:
        print("[Visualizar] matplotlib no disponible. Instala con: pip install matplotlib")
        return None

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ocupaciones   = [(1 - ind.aptitud[0]) * 100 for ind in frente_pareto]
    retrasos = [ind.aptitud[1]              for ind in frente_pareto]
    emergencias = [(1 - ind.aptitud[2]) * 100  for ind in frente_pareto]

    sc = ax.scatter(ocupaciones, retrasos, emergencias, c=emergencias, cmap="plasma", s=60, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Tasa Admisión Emergencias (%)")

    ax.set_xlabel("Tasa Ocupación (%)", labelpad=10)
    ax.set_ylabel("Índice Retraso", labelpad=10)
    ax.set_zlabel("Tasa Emergencias (%)", labelpad=10)
    ax.set_title("Frente de Pareto - UCI NSGA-II\nHospital Regional Lambayeque", pad=15)

    ruta = os.path.join(out_dir, "frente_pareto_3d.png")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizar] Gráfico Pareto guardado: {ruta}")
    return ruta


def graficar_convergencia(historial: list[dict], out_dir: str = ".") -> Optional[str]:
    """Grafica la convergencia de las métricas a lo largo de las generaciones."""
    if not HAS_MPL or not historial:
        return None

    generaciones   = [h["generación"]  for h in historial]
    ocupaciones   = [h.get("mejor_ocupación", 0) * 100 for h in historial]
    retrasos = [h.get("mejor_retraso", 0) for h in historial]
    emergencias = [h.get("mejor_emergencia", 0) * 100 for h in historial]
    tamaño_pareto = [h.get("tamaño_pareto", 0) for h in historial]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Convergencia NSGA-II - UCI Hospital Lambayeque", fontsize=13)

    axes[0, 0].plot(generaciones, ocupaciones, color="#2196F3")
    axes[0, 0].set_title("Mejor Tasa Ocupación (%)")
    axes[0, 0].set_xlabel("Generación")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(generaciones, retrasos, color="#F44336")
    axes[0, 1].set_title("Mejor Índice Retraso (mín)")
    axes[0, 1].set_xlabel("Generación")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(generaciones, emergencias, color="#4CAF50")
    axes[1, 0].set_title("Mejor Tasa Admisión Emergencias (%)")
    axes[1, 0].set_xlabel("Generación")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(generaciones, tamaño_pareto, color="#FF9800")
    axes[1, 1].set_title("Tamaño Frente de Pareto")
    axes[1, 1].set_xlabel("Generación")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(out_dir, "convergencia.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizar] Gráfico convergencia guardado: {ruta}")
    return ruta


def graficar_mapa_calor_ocupacion(
    individuo: Individuo,
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
    n_camas = 28
    grilla   = [["" for _ in range(HORIZONTE_PLANIFICACIÓN)] for _ in range(n_camas)]

    for día, lista_pacientes in individuo.programa.items():
        for p in lista_pacientes:
            if p.cama_asignada is not None:
                bid = p.cama_asignada - 1  # 0-indexed
                if 0 <= bid < n_camas:
                    grilla[bid][día - 1] = p.categoría

    # Mapeo de categoría a valor numérico
    mapa_categorías = {"actual": 1, "emergencia": 2, "electivo": 3, "": 0}
    datos = [[mapa_categorías.get(cell, 0) for cell in row] for row in grilla]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.colors.ListedColormap(["#ECEFF1", "#81C784", "#64B5F6", "#FFB74D"])
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(datos, cmap=cmap, norm=norm, aspect="auto")

    # Etiquetas de ejes
    ax.set_xticks(range(HORIZONTE_PLANIFICACIÓN))
    ax.set_xticklabels([f"Día {d+1}" for d in range(HORIZONTE_PLANIFICACIÓN)])
    ax.set_yticks(range(n_camas))

    # Etiquetas de camas con tipo
    etiquetas_camas = []
    for bid in range(1, n_camas + 1):
        from config import MAPA_TIPO_CAMA
        tipo_cama = MAPA_TIPO_CAMA.get(bid, "?")
        abr_tipo = {"adulto": "A", "pediátrico": "P", "neonatal": "N"}.get(tipo_cama, "?")
        etiquetas_camas.append(f"Cama {bid} ({abr_tipo})")
    ax.set_yticklabels(etiquetas_camas, fontsize=7)

    # Leyenda
    elementos_leyenda = [
        mpatches.Patch(color="#ECEFF1", label="Vacía"),
        mpatches.Patch(color="#81C784", label="Paciente Actual"),
        mpatches.Patch(color="#64B5F6", label="Emergencia"),
        mpatches.Patch(color="#FFB74D", label="Electivo"),
    ]
    ax.legend(handles=elementos_leyenda, loc="upper right", fontsize=8)

    # Separadores por tipo de cama
    ax.axhline(y=17.5, color="black", linewidth=1.5, linestyle="--")
    ax.axhline(y=21.5, color="black", linewidth=1.5, linestyle="--")
    ax.text(HORIZONTE_PLANIFICACIÓN + 0.1, 9,   "ADULTO",    fontsize=7, rotation=90, va="center")
    ax.text(HORIZONTE_PLANIFICACIÓN + 0.1, 19.5,"PED.",      fontsize=7, rotation=90, va="center")
    ax.text(HORIZONTE_PLANIFICACIÓN + 0.1, 25,  "NEONATAL",  fontsize=7, rotation=90, va="center")

    ax.set_title(f"{title}\n(Solución Pareto - mejor ocupación)", fontsize=11)
    plt.tight_layout()
    ruta = os.path.join(out_dir, "mapa_calor_camas.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizar] Heatmap de camas guardado: {ruta}")
    return ruta
