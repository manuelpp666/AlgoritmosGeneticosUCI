"""
main.py — Punto de entrada del sistema AG-UCI.

Hospital Regional Lambayeque – Optimización de Asignación de Camas UCI

Uso:
    python main.py                    # 40 pacientes, semilla aleatoria
    python main.py --pacientes 50     # 50 pacientes en lista de espera
    python main.py --semilla 42       # resultado reproducible
    python main.py --generaciones 200 # más generaciones
    python main.py --sin-graficas     # omitir matplotlib
"""

import argparse
import time
import sys
import os

# Asegurar que el directorio actual esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacientes  import generar_lista_espera, mostrar_resumen
from genetico   import ejecutar_ag
from resultados import (
    decodificar_solucion, imprimir_reporte,
    graficar_evolucion, graficar_gantt_camas, graficar_ocupacion_diaria,
)
from config import NUM_GENERACIONES, TAMANIO_POBLACION


def barra_progreso(gen: int, total: int, mejor: float, avg: float) -> None:
    """Muestra progreso en consola con barra animada."""
    pct    = gen / total
    ancho  = 35
    lleno  = int(ancho * pct)
    barra  = "█" * lleno + "░" * (ancho - lleno)
    print(
        f"\r  Gen {gen:>4}/{total}  [{barra}]  "
        f"Mejor: {mejor:>9.2f}  Avg: {avg:>9.2f}  ",
        end="", flush=True
    )
    if gen == total:
        print()  # salto de línea al terminar


def main():
    parser = argparse.ArgumentParser(
        description="AG-UCI: Optimización de Asignación de Camas UCI – HRL"
    )
    parser.add_argument("--pacientes",     type=int,  default=40,
                        help="Número de pacientes en lista de espera (default: 40)")
    parser.add_argument("--semilla",       type=int,  default=None,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--generaciones",  type=int,  default=NUM_GENERACIONES,
                        help=f"Número de generaciones (default: {NUM_GENERACIONES})")
    parser.add_argument("--poblacion",     type=int,  default=TAMANIO_POBLACION,
                        help=f"Tamaño de población (default: {TAMANIO_POBLACION})")
    parser.add_argument("--sin-graficas",  action="store_true",
                        help="Omitir la generación de gráficas matplotlib")
    args = parser.parse_args()

    # Actualizar parámetros si se pasan por argumento
    import config as cfg
    cfg.NUM_GENERACIONES   = args.generaciones
    cfg.TAMANIO_POBLACION  = args.poblacion

    print("\n" + "═" * 70)
    print("  🏥  AG-UCI – Algoritmo Genético para Asignación de Camas UCI")
    print("       Hospital Regional Lambayeque")
    print("═" * 70)
    print(f"  Pacientes en lista : {args.pacientes}")
    print(f"  Generaciones       : {args.generaciones}")
    print(f"  Tamaño población   : {args.poblacion}")
    print(f"  Semilla            : {args.semilla if args.semilla else 'aleatoria'}")
    print("═" * 70)

    # ── 1. Generar lista de espera ────────────────────────────────
    print("\n📥 Generando lista de espera de pacientes...")
    pacientes = generar_lista_espera(n_pacientes=args.pacientes, semilla=args.semilla)
    mostrar_resumen(pacientes)

    # ── 2. Ejecutar Algoritmo Genético ────────────────────────────
    print(f"\n🧬 Iniciando Algoritmo Genético ({args.generaciones} generaciones)...\n")
    t_inicio = time.time()

    total_gen = args.generaciones
    def cb(gen, mejor, avg):
        barra_progreso(gen, total_gen, mejor, avg)

    mejor_cromosoma, mejor_fitness, historial = ejecutar_ag(
        pacientes=pacientes,
        callback_generacion=cb,
    )

    t_total = time.time() - t_inicio
    print(f"\n  ✅ Optimización completada en {t_total:.2f} s")
    print(f"  🏆 Mejor fitness alcanzado: {mejor_fitness:,.2f}")

    # ── 3. Decodificar y reportar solución ────────────────────────
    print("\n📋 Decodificando solución óptima...\n")
    solucion = decodificar_solucion(mejor_cromosoma, pacientes)
    imprimir_reporte(solucion, mejor_fitness)

    # ── 4. Gráficas ───────────────────────────────────────────────
    if not args.sin_graficas:
        print("\n📊 Generando visualizaciones...")
        graficar_evolucion(historial,       ruta_salida="evolucion_fitness.png")
        graficar_gantt_camas(solucion,      ruta_salida="gantt_camas.png")
        graficar_ocupacion_diaria(solucion, ruta_salida="ocupacion_diaria.png")
        print("\n  Gráficas guardadas en el directorio actual.")
    else:
        print("\n  ⏭️  Gráficas omitidas (--sin-graficas).")

    print("\n" + "═" * 70)
    print("  Ejecución finalizada. ¡Éxito!")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
