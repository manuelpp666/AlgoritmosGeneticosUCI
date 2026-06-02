"""
resultados.py — Decodificación del cromosoma óptimo y generación de reportes.
"""

from pacientes import Paciente
from config import CAMAS, capacidad_dia, MEDICOS_POR_DIA, DIAS_SEMANA, DIA_NO_ADMITIDO
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NOMBRE_DIAS = {1:"Lunes", 2:"Martes", 3:"Miércoles",
               4:"Jueves", 5:"Viernes", 6:"Sábado", 7:"Domingo"}

COLOR_TIPO = {
    "Adulto":     "#2196F3",   # azul
    "Pediatrica": "#4CAF50",   # verde
    "Neonatal":   "#FF9800",   # naranja
}

COLOR_GRAVEDAD = {
    "Alta":   "#F44336",  # rojo
    "Media":  "#FFC107",  # amarillo
    "Baja":   "#4CAF50",  # verde
}


def _nivel_gravedad(g: float) -> str:
    if g >= 0.80:
        return "Alta"
    elif g >= 0.55:
        return "Media"
    return "Baja"


def decodificar_solucion(cromosoma: list[tuple[int, int]],
                          pacientes: list[Paciente]) -> dict:
    """
    Transforma el cromosoma en un plan de asignación legible.

    Retorna un diccionario con:
      - 'admitidos': lista de asignaciones válidas
      - 'no_admitidos': pacientes con dia=0
      - 'violaciones': lista de descripciones de restricciones violadas
      - 'estadisticas': resumen numérico
    """
    admitidos    = []
    no_admitidos = []
    violaciones  = []

    # Mapa de ocupación por cama
    ocupacion: dict[int, list[dict]] = {k: [] for k in CAMAS}
    admitidos_por_dia: dict[int, list] = {d: [] for d in range(1, DIAS_SEMANA + 1)}

    for i, (dia, cama) in enumerate(cromosoma):
        p = pacientes[i]

        if dia == DIA_NO_ADMITIDO:
            no_admitidos.append({"paciente": p, "motivo": "No admitido (día=0)"})
            continue

        tipo_cama  = CAMAS.get(cama, "?")
        compatible = tipo_cama == p.especialidad_requerida
        dia_salida = dia + p.los_estimado

        registro = {
            "paciente":    p,
            "dia_ingreso": dia,
            "dia_salida":  dia_salida,
            "cama_id":     cama,
            "tipo_cama":   tipo_cama,
            "compatible":  compatible,
            "retraso":     max(0, dia - p.fecha_esperada),
        }
        admitidos.append(registro)
        ocupacion[cama].append(registro)
        admitidos_por_dia[dia].append(registro)

        if not compatible:
            violaciones.append(
                f"⚠️  Paciente {p.nombre} ({p.especialidad_requerida}) → "
                f"Cama {cama} ({tipo_cama}) – INCOMPATIBLE"
            )

    # Verificar solapamientos
    for cama_id, periodos in ocupacion.items():
        periodos_ord = sorted(periodos, key=lambda x: x["dia_ingreso"])
        for k in range(len(periodos_ord) - 1):
            a = periodos_ord[k]
            b = periodos_ord[k + 1]
            if b["dia_ingreso"] <= a["dia_salida"]:
                violaciones.append(
                    f"⚠️  Solapamiento Cama {cama_id}: "
                    f"{a['paciente'].nombre} (sale día {a['dia_salida']}) "
                    f"← {b['paciente'].nombre} (entra día {b['dia_ingreso']})"
                )

    # Verificar capacidad médica diaria
    for dia in range(1, DIAS_SEMANA + 1):
        cap    = capacidad_dia(dia)
        total  = len(admitidos_por_dia[dia])
        medicos = MEDICOS_POR_DIA[dia]
        if total > cap:
            violaciones.append(
                f"⚠️  Día {NOMBRE_DIAS[dia]}: {total} pacientes > cap. {cap} "
                f"({medicos} médicos)"
            )

    # Estadísticas
    total_pac   = len(pacientes)
    total_adm   = len(admitidos)
    graves_adm  = sum(1 for r in admitidos if r["paciente"].gravedad >= 0.80)
    graves_tot  = sum(1 for p in pacientes if p.gravedad >= 0.80)
    incompat    = sum(1 for r in admitidos if not r["compatible"])
    solapam     = sum(1 for v in violaciones if "Solapamiento" in v)
    cap_violada = sum(1 for v in violaciones if "médicos" in v or "cap." in v)

    estadisticas = {
        "total_pacientes":   total_pac,
        "admitidos":         total_adm,
        "no_admitidos":      total_pac - total_adm,
        "tasa_admision":     total_adm / total_pac * 100 if total_pac else 0,
        "graves_atendidos":  graves_adm,
        "graves_total":      graves_tot,
        "tasa_graves":       graves_adm / graves_tot * 100 if graves_tot else 0,
        "incompatibilidades": incompat,
        "solapamientos":     solapam,
        "violaciones_cap":   cap_violada,
        "total_violaciones": len(violaciones),
    }

    return {
        "admitidos":   admitidos,
        "no_admitidos": no_admitidos,
        "violaciones":  violaciones,
        "estadisticas": estadisticas,
        "admitidos_por_dia": admitidos_por_dia,
    }


# ══════════════════════════════════════════════════════════════════
#  IMPRESIÓN EN CONSOLA
# ══════════════════════════════════════════════════════════════════

def imprimir_reporte(solucion: dict, fitness: float) -> None:
    """Imprime el reporte completo en consola."""
    est  = solucion["estadisticas"]
    adm  = solucion["admitidos"]
    viols = solucion["violaciones"]

    print("\n" + "═" * 95)
    print("  PLAN ÓPTIMO DE ASIGNACIÓN UCI – Hospital Regional Lambayeque")
    print(f"  Fitness final: {fitness:,.2f}")
    print("═" * 95)

    # ── Tabla de admitidos ────────────────────────────────────────
    filas = []
    for r in sorted(adm, key=lambda x: (x["dia_ingreso"], x["cama_id"])):
        p = r["paciente"]
        comp_str = "✓" if r["compatible"] else "✗ INCOMPAT"
        filas.append([
            NOMBRE_DIAS[r["dia_ingreso"]],
            r["cama_id"],
            r["tipo_cama"],
            p.nombre,
            p.especialidad_requerida,
            f"{p.gravedad:.3f} ({_nivel_gravedad(p.gravedad)})",
            p.los_estimado,
            r["dia_salida"],
            f"+{r['retraso']}" if r["retraso"] else "0",
            comp_str,
        ])

    headers = ["Día Ingreso", "Cama", "Tipo Cama", "Paciente",
               "Espec.Req.", "Gravedad", "LOS", "Día Salida",
               "Retraso", "Compat."]
    print("\n📋 ASIGNACIONES:")
    print(tabulate(filas, headers=headers, tablefmt="rounded_grid"))

    # ── No admitidos ──────────────────────────────────────────────
    if solucion["no_admitidos"]:
        print(f"\n❌ NO ADMITIDOS ({len(solucion['no_admitidos'])}):")
        for na in solucion["no_admitidos"]:
            p = na["paciente"]
            print(f"   • {p.nombre} | {p.especialidad_requerida} | "
                  f"Gravedad: {p.gravedad:.3f} | LOS: {p.los_estimado} días")

    # ── Resumen por día ───────────────────────────────────────────
    print("\n📅 OCUPACIÓN POR DÍA:")
    filas_dia = []
    for dia in range(1, DIAS_SEMANA + 1):
        cap     = capacidad_dia(dia)
        medicos = MEDICOS_POR_DIA[dia]
        total   = len(solucion["admitidos_por_dia"][dia])
        estado  = "✓ OK" if total <= cap else f"✗ EXCEDE ({total - cap})"
        filas_dia.append([NOMBRE_DIAS[dia], medicos, cap, total,
                          f"{total/cap*100:.0f}%" if cap else "—", estado])
    print(tabulate(filas_dia,
                   headers=["Día", "Médicos", "Cap.Máx", "Admitidos", "Ocupación", "Estado"],
                   tablefmt="rounded_grid"))

    # ── Estadísticas globales ─────────────────────────────────────
    print(f"""
📊 ESTADÍSTICAS GLOBALES:
   Pacientes totales   : {est['total_pacientes']}
   Admitidos           : {est['admitidos']}  ({est['tasa_admision']:.1f}%)
   No admitidos        : {est['no_admitidos']}
   Graves atendidos    : {est['graves_atendidos']} / {est['graves_total']}  ({est['tasa_graves']:.1f}%)
   Incompatibilidades  : {est['incompatibilidades']}
   Solapamientos       : {est['solapamientos']}
   Violaciones totales : {est['total_violaciones']}
""")

    if viols:
        print("⚠️  VIOLACIONES DETECTADAS:")
        for v in viols:
            print(f"   {v}")
    else:
        print("✅ SOLUCIÓN SIN VIOLACIONES DE RESTRICCIONES")


# ══════════════════════════════════════════════════════════════════
#  VISUALIZACIONES MATPLOTLIB
# ══════════════════════════════════════════════════════════════════

def graficar_evolucion(historial: list[float], ruta_salida: str = "evolucion_fitness.png") -> None:
    """Gráfica de evolución del fitness a lo largo de las generaciones."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(historial, color="#1565C0", linewidth=2, label="Mejor Fitness")
    ax.fill_between(range(len(historial)), historial, alpha=0.15, color="#1565C0")
    ax.set_title("Evolución del Fitness – Algoritmo Genético UCI", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"   → Gráfica guardada: {ruta_salida}")


def graficar_gantt_camas(solucion: dict,
                          ruta_salida: str = "gantt_camas.png") -> None:
    """
    Diagrama de Gantt de ocupación de camas por semana.
    Cada barra = estancia de un paciente.
    Color = especialidad de la cama.
    """
    adm = solucion["admitidos"]
    if not adm:
        print("   ⚠️  Sin datos para graficar Gantt.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    camas_usadas = sorted(set(r["cama_id"] for r in adm))
    y_pos = {cama: i for i, cama in enumerate(camas_usadas)}

    for r in adm:
        cama_id   = r["cama_id"]
        dia_ini   = r["dia_ingreso"]
        duracion  = r["paciente"].los_estimado
        tipo_cama = r["tipo_cama"]
        color     = COLOR_TIPO.get(tipo_cama, "#9E9E9E")
        alpha     = 0.55 + r["paciente"].gravedad * 0.45  # más grave = más opaco

        bar = ax.barh(
            y=y_pos[cama_id],
            width=duracion,
            left=dia_ini - 0.5,
            height=0.6,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.8,
        )
        # Etiqueta con nombre del paciente
        ax.text(
            dia_ini - 0.5 + duracion / 2,
            y_pos[cama_id],
            r["paciente"].nombre.split("-")[0],
            ha="center", va="center",
            fontsize=6, color="white", fontweight="bold"
        )

    # Ejes
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"Cama {c}" for c in camas_usadas], fontsize=8)
    ax.set_xticks(range(1, DIAS_SEMANA + 2))
    ax.set_xticklabels(
        [f"Día {d}\n{NOMBRE_DIAS.get(d, '')}" for d in range(1, DIAS_SEMANA + 2)],
        fontsize=8
    )
    ax.set_xlim(0.5, DIAS_SEMANA + 0.5)
    ax.set_title("Gantt de Ocupación de Camas UCI (Plan Semanal Óptimo)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Día de la Semana")
    ax.set_ylabel("Cama UCI")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Leyenda
    parches = [
        mpatches.Patch(color=v, label=k) for k, v in COLOR_TIPO.items()
    ]
    ax.legend(handles=parches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"   → Gantt guardado: {ruta_salida}")


def graficar_ocupacion_diaria(solucion: dict,
                               ruta_salida: str = "ocupacion_diaria.png") -> None:
    """Barras de ocupación diaria vs capacidad máxima."""
    dias = list(range(1, DIAS_SEMANA + 1))
    ocupados = [len(solucion["admitidos_por_dia"][d]) for d in dias]
    caps     = [capacidad_dia(d) for d in dias]
    nombres  = [NOMBRE_DIAS[d] for d in dias]

    x = np.arange(len(dias))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, ocupados, width=0.5, label="Pacientes admitidos",
                  color=["#E53935" if o > c else "#43A047"
                         for o, c in zip(ocupados, caps)],
                  zorder=3)
    ax.plot(x, caps, color="#1565C0", marker="D", linewidth=2,
            label="Capacidad máxima (médicos)", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_title("Ocupación Diaria vs Capacidad Médica UCI", fontsize=13, fontweight="bold")
    ax.set_ylabel("Número de pacientes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, ocupados):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"   → Ocupación diaria guardada: {ruta_salida}")
