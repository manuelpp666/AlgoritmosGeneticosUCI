"""
nsga2.py - Núcleo del algoritmo NSGA-II
Non-dominated sorting, crowding distance, selection, crossover, mutación
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
from typing import Callable

from config import (
    MAX_BEDS_ACTIVE, PLANNING_HORIZON,
    CROSSOVER_RATE, MUTATION_RATE,
    BEDS,
)
from patients import Patient
from chromosome import Individual, assign_bed, beds_for_type


# ══════════════════════════════════════════════
#  NON-DOMINATED SORTING
# ══════════════════════════════════════════════
def dominates(a: Individual, b: Individual) -> bool:
    """
    a domina a b si a es al menos tan bueno en todos los objetivos
    y estrictamente mejor en al menos uno. (minimización)
    """
    fa, fb = a.fitness, b.fitness
    at_least_equal = all(fa[i] <= fb[i] for i in range(3))
    strictly_better = any(fa[i] <  fb[i] for i in range(3))
    return at_least_equal and strictly_better


def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    """
    Algoritmo de ordenamiento no-dominado O(MN²).
    Retorna lista de frentes de Pareto [F1, F2, ...].
    """
    n = len(population)
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # S_p
    dom_count:    list[int]       = [0]  * n                 # n_p
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    population[j].rank = current_front + 2
                    next_front.append(j)
        fronts.append(next_front)
        current_front += 1

    # Convertir índices a individuos
    result = []
    for f in fronts:
        if f:
            result.append([population[i] for i in f])
    return result


# ══════════════════════════════════════════════
#  CROWDING DISTANCE
# ══════════════════════════════════════════════
def crowding_distance_assignment(front: list[Individual]) -> None:
    """Asigna distancia de hacinamiento a cada individuo del frente."""
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding = 0.0

    n_obj = 3
    for m in range(n_obj):
        front.sort(key=lambda x: x.fitness[m])
        front[0].crowding   = float("inf")
        front[-1].crowding  = float("inf")
        f_min = front[0].fitness[m]
        f_max = front[-1].fitness[m]
        denom = f_max - f_min if f_max != f_min else 1e-10
        for i in range(1, n - 1):
            front[i].crowding += (
                (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / denom
            )


# ══════════════════════════════════════════════
#  SELECCIÓN BINARIA POR TORNEO
# ══════════════════════════════════════════════
def crowded_tournament(a: Individual, b: Individual) -> Individual:
    """Selección por torneo usando rank y crowding (NSGA-II)."""
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding >= b.crowding else b


def tournament_selection(
    population: list[Individual],
    rng: random.Random,
) -> Individual:
    """Selección por torneo binario."""
    i, j = rng.sample(range(len(population)), 2)
    return crowded_tournament(population[i], population[j])


# ══════════════════════════════════════════════
#  CROSSOVER
# ══════════════════════════════════════════════
def _count_beds_day(
    day: int,
    electives_day: list[Patient],
    carry_forward: dict[int, list[Patient]],
) -> int:
    """
    Cuenta camas ocupadas para un día dado considerando los pacientes
    arrastrados + los electivos programados ese día.
    """
    return len(carry_forward.get(day, [])) + len(electives_day)


def crossover(
    parent1: Individual,
    parent2: Individual,
    current_patients: list[Patient],
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """
    Crossover de fechas de admisión entre dos pacientes electivos.
    Intercambia el día de ingreso de dos electivos aleatorios (uno de cada padre)
    y verifica que no se supere MAX_BEDS_ACTIVE.
    """
    if rng.random() > CROSSOVER_RATE:
        return parent1.clone(), parent2.clone()

    child1 = parent1.clone()
    child2 = parent2.clone()

    # Recopilar todos los electivos programados en cada hijo
    def get_all_electives(ind: Individual) -> list[tuple[int, int]]:
        """Retorna lista de (day, idx_en_schedule_day) para cada electivo."""
        result = []
        for day, plist in ind.schedule.items():
            for idx in range(len(plist)):
                result.append((day, idx))
        return result

    elec1 = get_all_electives(child1)
    elec2 = get_all_electives(child2)

    if not elec1 or not elec2:
        return child1, child2

    # Elegir un electivo de cada hijo
    d1, i1 = rng.choice(elec1)
    d2, i2 = rng.choice(elec2)

    p1 = child1.schedule[d1][i1].clone()
    p2 = child2.schedule[d2][i2].clone()

    # Intentar intercambio en child1: reemplazar p1 (en d1) por p2 (en d2)
    if _is_swap_feasible(child1, d1, i1, p2, d2):
        child1.schedule[d1][i1] = p2.clone()
        child1.schedule[d1][i1].admission_day = d1

    # Intentar intercambio en child2: reemplazar p2 (en d2) por p1 (en d1)
    if _is_swap_feasible(child2, d2, i2, p1, d1):
        child2.schedule[d2][i2] = p1.clone()
        child2.schedule[d2][i2].admission_day = d2

    return child1, child2


def _is_swap_feasible(
    ind: Individual,
    day: int,
    idx: int,
    new_patient: Patient,
    original_day: int,
) -> bool:
    """
    Verifica que al insertar new_patient en `day` no se supere MAX_BEDS_ACTIVE.
    También verifica compatibilidad de tipo de cama.
    """
    # Comprobar que hay cama del tipo correcto disponible en ese día
    occupied_types = {p.patient_type for p in ind.schedule[day]}
    available = sum(
        1 for bid in beds_for_type(new_patient.patient_type)
        if bid <= MAX_BEDS_ACTIVE
    )
    used_of_type = sum(
        1 for p in ind.schedule[day]
        if p.patient_type == new_patient.patient_type
    )
    if used_of_type >= available:
        return False

    # Verificar límite total del día
    total_day = sum(len(v) for v in ind.schedule.values() if v)
    # Aproximación simple: si el día ya tiene muchos pacientes, rechazar
    if len(ind.schedule[day]) >= MAX_BEDS_ACTIVE:
        return False

    return True


# ══════════════════════════════════════════════
#  MUTACIÓN
# ══════════════════════════════════════════════
def mutation(
    individual: Individual,
    elective_waitlist: list[Patient],
    rng: random.Random,
) -> Individual:
    """
    Mutación: elimina un electivo programado e intenta insertar otro
    de la lista de espera con LOS ≤ al eliminado.
    """
    if rng.random() > MUTATION_RATE:
        return individual

    mutant = individual.clone()

    # Recopilar electivos programados
    scheduled = [
        (day, idx)
        for day, plist in mutant.schedule.items()
        for idx in range(len(plist))
    ]
    if not scheduled:
        return mutant

    # Elegir un electivo al azar para eliminar
    day, idx = rng.choice(scheduled)
    removed = mutant.schedule[day].pop(idx)

    # Buscar candidato de reemplazo con LOS ≤ removido y tipo igual
    # (para respetar las camas por tipo)
    candidates = [
        p for p in elective_waitlist
        if p.los <= removed.los
        and p.patient_type == removed.patient_type
        and not _is_already_scheduled(p.patient_id, mutant)
    ]

    if candidates:
        # Prioriza mayor loss_of_chance (urgencia)
        candidates.sort(key=lambda p: -p.loss_of_chance)
        new_p = candidates[0].clone()
        new_p.admission_day = day
        new_p.delay = max(0, day - (new_p.scheduled_day or day))
        mutant.schedule[day].append(new_p)

    return mutant


def _is_already_scheduled(pid: str, ind: Individual) -> bool:
    for plist in ind.schedule.values():
        for p in plist:
            if p.patient_id == pid:
                return True
    return False


# ══════════════════════════════════════════════
#  REINSERCIÓN
# ══════════════════════════════════════════════
def reinsertion(
    individual: Individual,
    elective_waitlist: list[Patient],
    current_patients: list[Patient],
    rng: random.Random,
) -> Individual:
    """
    Si quedan camas vacías en algún día, intenta insertar electivos
    priorizando menor LOS y mayor loss_of_chance.
    """
    ind = individual.clone()

    # Calcular cuántas camas hay disponibles por tipo por día (aproximación)
    for day in range(1, PLANNING_HORIZON + 1):
        # Contar camas usadas por tipo en este día
        used_per_type: dict[str, int] = {"adult": 0, "pediatric": 0, "neonatal": 0}
        for p in ind.schedule[day]:
            used_per_type[p.patient_type] += 1
        # También contar actuales que arrastran (simplificación)
        for ptype, cap in [("adult", 18), ("pediatric", 4), ("neonatal", 6)]:
            max_type = min(cap, MAX_BEDS_ACTIVE)  # cap por tipo
            available = max_type - used_per_type[ptype]
            if available <= 0:
                continue

            # Candidatos no programados del tipo correcto
            candidates = sorted(
                [
                    p for p in elective_waitlist
                    if p.patient_type == ptype
                    and not _is_already_scheduled(p.patient_id, ind)
                ],
                key=lambda p: (p.los, -p.loss_of_chance),
            )
            inserted = 0
            for p in candidates:
                if inserted >= available:
                    break
                new_p = p.clone()
                new_p.admission_day = day
                new_p.delay = max(0, day - (new_p.scheduled_day or day))
                ind.schedule[day].append(new_p)
                inserted += 1

    return ind


# ══════════════════════════════════════════════
#  SELECCIÓN DE PRÓXIMA GENERACIÓN
# ══════════════════════════════════════════════
def select_next_generation(
    combined: list[Individual],
    pop_size: int,
) -> list[Individual]:
    """
    Merge + NSGA-II selection: llena la nueva población con los mejores frentes
    usando crowding distance como desempate en el frente límite.
    """
    fronts = fast_non_dominated_sort(combined)
    new_pop: list[Individual] = []

    for front in fronts:
        crowding_distance_assignment(front)
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            # Tomar los mejores por crowding distance hasta completar pop_size
            remaining = pop_size - len(new_pop)
            front.sort(key=lambda x: -x.crowding)
            new_pop.extend(front[:remaining])
            break

    return new_pop
