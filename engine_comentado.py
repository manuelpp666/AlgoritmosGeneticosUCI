"""
engine.py - Motor principal del NSGA-II
Orquesta la evolución completa y registra métricas
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
import time
from typing import Optional, Tuple

from config import (
    TAMAÑO_POBLACIÓN, GENERACIONES_MAX,
    TASA_CRUZAMIENTO, TASA_MUTACIÓN,
    SEMILLA_ALEATORIA,
)
from patients import (
    Paciente,
    generar_pacientes_actuales,
    generar_lista_espera_electivos,
)
from chromosome import Individuo
from nsga2 import (
    ordenamiento_no_dominado_rápido,
    asignación_distancia_hacinamiento,
    selección_torneo,
    cruzamiento,
    mutación,
    reinserción,
    seleccionar_próxima_generación,
)


# ══════════════════════════════════════════════
#  MOTOR NSGA-II
# ══════════════════════════════════════════════
class NSGA2_UCI:
    """
    Motor de optimización NSGA-II para asignación de camas UCI.
    """

    def __init__(
        self,
        pacientes_actuales: Optional[list[Paciente]] = None,
        lista_espera_electivos: Optional[list[Paciente]] = None,
        semilla: int = SEMILLA_ALEATORIA,
    ):
        self.rng = random.Random(semilla)

        # Datos hospitalarios
        if pacientes_actuales is None:
            self.pacientes_actuales = generar_pacientes_actuales(self.rng)
        else:
            self.pacientes_actuales = pacientes_actuales

        if lista_espera_electivos is None:
            self.lista_espera_electivos = generar_lista_espera_electivos(self.rng, n=40)
        else:
            self.lista_espera_electivos = lista_espera_electivos

        # Estado del algoritmo
        self.población:   list[Individuo] = []
        self.frente_pareto: list[Individuo] = []
        self.historial:      list[dict]       = []   # métricas por generación

    # ── Inicialización ────────────────────────
    def inicializar_población(self) -> None:
        print(f"[NSGA-II] Inicializando población de {TAMAÑO_POBLACIÓN} individuos...")
        t0 = time.time()
        self.población = [
            Individuo.crear_aleatorio(
                self.pacientes_actuales,
                self.lista_espera_electivos,
                self.rng,
            )
            for _ in range(TAMAÑO_POBLACIÓN)
        ]
        self._evaluar_población(self.población)
        frentes = ordenamiento_no_dominado_rápido(self.población)
        for frente in frentes:
            asignación_distancia_hacinamiento(frente)
        print(f"[NSGA-II] Población inicial lista en {time.time()-t0:.2f}s")

    # ── Evaluación ────────────────────────────
    def _evaluar_población(self, población: list[Individuo]) -> None:
        """Evalúa la aptitud de cada individuo en la población."""
        for individuo in población:
            # Cada individuo usa su propio rng seed derivado para reproducibilidad
            rng_evaluación = random.Random(self.rng.randint(0, 2**31))
            individuo.evaluar(self.pacientes_actuales, self.lista_espera_electivos, rng_evaluación)

    # ── Un ciclo evolutivo ────────────────────
    def _evolucionar_una_generación(self) -> None:
        """Realiza un ciclo completo de evolución: selección, cruzamiento, mutación."""
        descendencia: list[Individuo] = []

        while len(descendencia) < TAMAÑO_POBLACIÓN:
            # Selección
            p1 = selección_torneo(self.población, self.rng)
            p2 = selección_torneo(self.población, self.rng)

            # Cruzamiento
            c1, c2 = cruzamiento(
                p1, p2, self.pacientes_actuales, self.rng
            )

            # Mutación
            c1 = mutación(c1, self.lista_espera_electivos, self.rng)
            c2 = mutación(c2, self.lista_espera_electivos, self.rng)

            # Reinserción
            c1 = reinserción(c1, self.lista_espera_electivos, self.pacientes_actuales, self.rng)
            c2 = reinserción(c2, self.lista_espera_electivos, self.pacientes_actuales, self.rng)

            descendencia.extend([c1, c2])

        # Evaluar hijos
        self._evaluar_población(descendencia)

        # Merge + selección NSGA-II
        combinada = self.población + descendencia
        self.población = seleccionar_próxima_generación(combinada, TAMAÑO_POBLACIÓN)

    # ── Estadísticas de generación ────────────
    def _registrar_estadísticas(self, generación: int) -> dict:
        """Registra métricas de la generación actual."""
        pareto = [ind for ind in self.población if ind.rango == 1]
        if not pareto:
            return {}

        f1_vals = [1.0 - ind.aptitud[0] for ind in pareto]  # ocupación real
        f2_vals = [ind.aptitud[1] for ind in pareto]         # retraso
        f3_vals = [1.0 - ind.aptitud[2] for ind in pareto]  # tasa emergencias

        estadísticas = {
            "generación": generación,
            "tamaño_pareto": len(pareto),
            "mejor_ocupación": max(f1_vals),
            "ocupación_promedio": sum(f1_vals) / len(f1_vals),
            "mejor_retraso": min(f2_vals),
            "retraso_promedio": sum(f2_vals) / len(f2_vals),
            "mejor_emergencia": max(f3_vals),
            "emergencia_promedio": sum(f3_vals) / len(f3_vals),
        }
        return estadísticas

    # ── Ejecución principal ───────────────────
    def ejecutar(self, verboso: bool = True) -> list[Individuo]:
        """
        Ejecuta NSGA-II hasta GENERACIONES_MAX.
        Retorna el frente de Pareto final.
        """
        self.inicializar_población()

        for generación in range(1, GENERACIONES_MAX + 1):
            t0 = time.time()
            self._evolucionar_una_generación()
            estadísticas = self._registrar_estadísticas(generación)
            self.historial.append(estadísticas)

            if verboso and generación % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"  Gen {generación:>4}/{GENERACIONES_MAX} | "
                    f"Pareto: {estadísticas.get('tamaño_pareto', 0):>3} | "
                    f"Ocupación: {estadísticas.get('mejor_ocupación', 0):.3f} | "
                    f"Retraso: {estadísticas.get('mejor_retraso', 0):.2f} | "
                    f"Emergencia: {estadísticas.get('mejor_emergencia', 0):.3f} | "
                    f"{elapsed:.2f}s"
                )

        self.frente_pareto = [ind for ind in self.población if ind.rango == 1]
        print(f"\n[NSGA-II] Finalizado. Frente de Pareto: {len(self.frente_pareto)} soluciones.")
        return self.frente_pareto

    # ── Estrategias extremas ──────────────────
    def obtener_3_estrategias_extremas(self) -> tuple[Optional[Individuo], Optional[Individuo], Optional[Individuo]]:
        """
        Extrae los 3 individuos más representativos del Frente de Pareto:
        1. Máxima ocupación (minimiza f1)
        2. Máxima tasa de emergencias (minimiza f3)
        3. Más equilibrado (compromiso entre objetivos)
        
        Retorna: (mejor_ocupación, mejor_emergencias, equilibrado)
        """
        if not self.frente_pareto:
            return None, None, None

        # 1. Mejor ocupación: minimizar f1 (1 - ocupación)
        mejor_ocup = min(self.frente_pareto, key=lambda x: x.aptitud[0])

        # 2. Mejor emergencias: minimizar f3 (1 - tasa_emergencias)
        mejor_emerg = min(self.frente_pareto, key=lambda x: x.aptitud[2])

        # 3. Más equilibrado: minimizar suma ponderada (o distancia euclidiana)
        mejor_equilib = min(
            self.frente_pareto,
            key=lambda x: (
                0.3 * x.aptitud[0] +
                0.3 * x.aptitud[1] +
                0.4 * x.aptitud[2]
            )
        )

        # Evitar repeticiones
        estrategias = [mejor_ocup]
        if mejor_emerg not in estrategias:
            estrategias.append(mejor_emerg)
        if mejor_equilib not in estrategias and len(estrategias) < 3:
            estrategias.append(mejor_equilib)

        # Si hay repeticiones, llenar con siguientes mejores
        while len(estrategias) < 3 and len(self.frente_pareto) > len(estrategias):
            for ind in sorted(self.frente_pareto, key=lambda x: sum(x.aptitud)):
                if ind not in estrategias:
                    estrategias.append(ind)
                    break

        return tuple(estrategias) if len(estrategias) >= 1 else (None, None, None)
