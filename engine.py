"""
engine.py - Motor principal del NSGA-II
Orquesta la evolución completa y registra métricas
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
import time
from typing import Optional

from config import (
    POPULATION_SIZE, MAX_GENERATIONS,
    CROSSOVER_RATE, MUTATION_RATE,
    RANDOM_SEED,
)
from patients import (
    Patient,
    generate_current_patients,
    generate_elective_waitlist,
)
from chromosome import Individual
from nsga2 import (
    fast_non_dominated_sort,
    crowding_distance_assignment,
    tournament_selection,
    crossover,
    mutation,
    reinsertion,
    select_next_generation,
)


# ══════════════════════════════════════════════
#  MOTOR NSGA-II
# ══════════════════════════════════════════════
class NSGAII_UCI:
    """
    Motor de optimización NSGA-II para asignación de camas UCI.
    """

    def __init__(
        self,
        current_patients: Optional[list[Patient]] = None,
        elective_waitlist: Optional[list[Patient]] = None,
        seed: int = RANDOM_SEED,
    ):
        self.rng = random.Random(seed)

        # Datos hospitalarios
        if current_patients is None:
            self.current_patients = generate_current_patients(self.rng)
        else:
            self.current_patients = current_patients

        if elective_waitlist is None:
            self.elective_waitlist = generate_elective_waitlist(self.rng, n=40)
        else:
            self.elective_waitlist = elective_waitlist

        # Estado del algoritmo
        self.population:   list[Individual] = []
        self.pareto_front: list[Individual] = []
        self.history:      list[dict]       = []   # métricas por generación

    # ── Inicialización ────────────────────────
    def initialize_population(self) -> None:
        print(f"[NSGA-II] Inicializando población de {POPULATION_SIZE} individuos...")
        t0 = time.time()
        self.population = [
            Individual.create_random(
                self.current_patients,
                self.elective_waitlist,
                self.rng,
            )
            for _ in range(POPULATION_SIZE)
        ]
        self._evaluate_population(self.population)
        fronts = fast_non_dominated_sort(self.population)
        for front in fronts:
            crowding_distance_assignment(front)
        print(f"[NSGA-II] Población inicial lista en {time.time()-t0:.2f}s")

    # ── Evaluación ────────────────────────────
    def _evaluate_population(self, pop: list[Individual]) -> None:
        for ind in pop:
            # Cada individuo usa su propio rng seed derivado para reproducibilidad
            eval_rng = random.Random(self.rng.randint(0, 2**31))
            ind.evaluate(self.current_patients, self.elective_waitlist, eval_rng)

    # ── Un ciclo evolutivo ────────────────────
    def _evolve_one_generation(self) -> None:
        offspring: list[Individual] = []

        while len(offspring) < POPULATION_SIZE:
            # Selección
            p1 = tournament_selection(self.population, self.rng)
            p2 = tournament_selection(self.population, self.rng)

            # Crossover
            c1, c2 = crossover(
                p1, p2, self.current_patients, self.rng
            )

            # Mutación
            c1 = mutation(c1, self.elective_waitlist, self.rng)
            c2 = mutation(c2, self.elective_waitlist, self.rng)

            # Reinserción
            c1 = reinsertion(c1, self.elective_waitlist, self.current_patients, self.rng)
            c2 = reinsertion(c2, self.elective_waitlist, self.current_patients, self.rng)

            offspring.extend([c1, c2])

        # Evaluar hijos
        self._evaluate_population(offspring)

        # Merge + selección NSGA-II
        combined = self.population + offspring
        self.population = select_next_generation(combined, POPULATION_SIZE)

    # ── Estadísticas de generación ────────────
    def _record_stats(self, gen: int) -> dict:
        pareto = [ind for ind in self.population if ind.rank == 1]
        if not pareto:
            return {}

        f1_vals = [1.0 - ind.fitness[0] for ind in pareto]  # ocupación real
        f2_vals = [ind.fitness[1] for ind in pareto]         # delay
        f3_vals = [1.0 - ind.fitness[2] for ind in pareto]  # tasa emergencias

        stats = {
            "generation":     gen,
            "pareto_size":    len(pareto),
            "best_occ":       max(f1_vals),
            "avg_occ":        sum(f1_vals) / len(f1_vals),
            "best_delay":     min(f2_vals),
            "avg_delay":      sum(f2_vals) / len(f2_vals),
            "best_emerg":     max(f3_vals),
            "avg_emerg":      sum(f3_vals) / len(f3_vals),
        }
        return stats

    # ── Ejecución principal ───────────────────
    def run(self, verbose: bool = True) -> list[Individual]:
        """
        Ejecuta NSGA-II hasta MAX_GENERATIONS.
        Retorna el frente de Pareto final.
        """
        self.initialize_population()

        for gen in range(1, MAX_GENERATIONS + 1):
            t0 = time.time()
            self._evolve_one_generation()
            stats = self._record_stats(gen)
            self.history.append(stats)

            if verbose and gen % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"  Gen {gen:>4}/{MAX_GENERATIONS} | "
                    f"Pareto: {stats.get('pareto_size', 0):>3} | "
                    f"Occ: {stats.get('best_occ', 0):.3f} | "
                    f"Delay: {stats.get('best_delay', 0):.2f} | "
                    f"Emerg: {stats.get('best_emerg', 0):.3f} | "
                    f"{elapsed:.2f}s"
                )

        self.pareto_front = [ind for ind in self.population if ind.rank == 1]
        print(f"\n[NSGA-II] Finalizado. Frente de Pareto: {len(self.pareto_front)} soluciones.")
        return self.pareto_front
