"""
chromosome.py - Cromosoma (Individuo) y motor de simulación diaria
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
import math
from copy import deepcopy
from typing import Optional

from config import (
    BEDS, BED_TYPE_MAP, MAX_BEDS_ACTIVE, TOTAL_BEDS,
    PLANNING_HORIZON, POISSON_LAMBDA,
)
from patients import Patient, generate_emergency_patients


# ──────────────────────────────────────────────
#  UTILIDADES
# ──────────────────────────────────────────────
def poisson_sample(lam: float, rng: random.Random) -> int:
    """Generación de muestra Poisson (Knuth)."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def beds_for_type(patient_type: str) -> list[int]:
    """Devuelve lista de IDs de cama compatibles con el tipo de paciente."""
    return list(BEDS[patient_type])


def assign_bed(patient: Patient, occupied: set[int]) -> Optional[int]:
    """
    Intenta asignar la primera cama libre del tipo correcto.
    Retorna el ID de cama o None si no hay disponible.
    """
    for bid in beds_for_type(patient.patient_type):
        if bid not in occupied and bid <= MAX_BEDS_ACTIVE:
            return bid
    return None


# ──────────────────────────────────────────────
#  CROMOSOMA
# ──────────────────────────────────────────────
class Individual:
    """
    Un individuo = plan semanal de 7 días para la UCI.

    Atributos:
        schedule  : dict[day → list[Patient]]   plan electivos por día
        fitness   : (f1, f2, f3) calculado tras evaluate()
        rank      : frente de Pareto
        crowding  : distancia de hacinamiento NSGA-II
    """

    def __init__(self):
        self.schedule: dict[int, list[Patient]] = {d: [] for d in range(1, PLANNING_HORIZON + 1)}
        self.fitness:  tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.rank:     int   = 0
        self.crowding: float = 0.0

    # ── Construcción ──────────────────────────
    @classmethod
    def create_random(
        cls,
        current_patients: list[Patient],
        elective_waitlist: list[Patient],
        rng: random.Random,
    ) -> "Individual":
        """
        Genera un individuo aleatorio respetando la jerarquía de asignación.
        """
        ind = cls()
        elective_pool = [p.clone() for p in elective_waitlist]
        rng.shuffle(elective_pool)

        # Puntero a los electivos aún no asignados
        elective_idx = 0

        # Estado de la UCI: pacientes activos que "arrastran" al día siguiente
        active_patients: list[Patient] = [p.clone() for p in current_patients]

        for day in range(1, PLANNING_HORIZON + 1):
            occupied: set[int] = set()

            # Paso 1: pacientes actuales (inamovibles)
            still_active: list[Patient] = []
            for p in active_patients:
                if p.remaining_los > 0:
                    bid = assign_bed(p, occupied)
                    if bid is not None:
                        p.assigned_bed = bid
                        occupied.add(bid)
                        still_active.append(p)
                    # Si no hay cama del tipo correcto disponible, caso excepcional ignorado

            # Paso 2: emergencias (Poisson)
            n_emerg = poisson_sample(POISSON_LAMBDA, rng)
            emergencies = generate_emergency_patients(rng, day, n_emerg)
            admitted_emerg: list[Patient] = []
            for ep in emergencies:
                bid = assign_bed(ep, occupied)
                if bid is not None:
                    ep.assigned_bed = bid
                    ep.admission_day = day
                    occupied.add(bid)
                    admitted_emerg.append(ep)

            # Paso 3: electivos (rellenar huecos)
            day_electives: list[Patient] = []
            while elective_idx < len(elective_pool):
                ep = elective_pool[elective_idx]
                bid = assign_bed(ep, occupied)
                if bid is not None:
                    ep_copy = ep.clone()
                    ep_copy.assigned_bed  = bid
                    ep_copy.admission_day = day
                    ep_copy.delay = max(0, day - (ep.scheduled_day or day))
                    occupied.add(bid)
                    day_electives.append(ep_copy)
                    elective_idx += 1
                else:
                    break  # No quedan camas disponibles

            ind.schedule[day] = day_electives

            # Actualizar estado para el día siguiente
            next_active: list[Patient] = []
            for p in still_active:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    next_active.append(p)
            for p in admitted_emerg:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    p.category = "current"
                    next_active.append(p)
            for p in day_electives:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    p.category = "current"
                    next_active.append(p)

            active_patients = next_active

        return ind

    # ── Evaluación (fitness) ──────────────────
    def evaluate(
        self,
        current_patients: list[Patient],
        elective_waitlist: list[Patient],
        rng: random.Random,
    ) -> None:
        """
        Simula la semana completa y calcula (f1, f2, f3).
        f1 = tasa ocupación  (maximizar → convertida en minimización: 1-f1)
        f2 = índice retraso  (minimizar)
        f3 = tasa admisión emergencias (maximizar → 1-f3)
        """
        total_beds_used   = 0
        total_delay_score = 0.0
        total_emerg_rate  = 0.0
        max_possible_emerg_total = 0

        active_patients: list[Patient] = [p.clone() for p in current_patients]

        # Reconstituimos un pool de electivos usando el schedule
        scheduled_ids = {
            day: [p.patient_id for p in plist]
            for day, plist in self.schedule.items()
        }

        for day in range(1, PLANNING_HORIZON + 1):
            occupied: set[int] = set()
            day_patients_count = 0

            # Paso 1: actuales
            still_active = []
            for p in active_patients:
                if p.remaining_los > 0:
                    bid = assign_bed(p, occupied)
                    if bid is not None:
                        occupied.add(bid)
                        day_patients_count += 1
                        still_active.append(p)

            # Paso 2: emergencias
            n_emerg = poisson_sample(POISSON_LAMBDA, rng)
            emergencies = generate_emergency_patients(rng, day, n_emerg)
            # Máximo posible según Poisson (usamos techo de 3*lambda como límite)
            max_possible = min(int(3 * POISSON_LAMBDA), MAX_BEDS_ACTIVE)
            max_possible_emerg_total += max_possible

            admitted_emerg = []
            for ep in emergencies:
                bid = assign_bed(ep, occupied)
                if bid is not None:
                    ep.assigned_bed = bid
                    ep.admission_day = day
                    occupied.add(bid)
                    day_patients_count += 1
                    admitted_emerg.append(ep)

            emerg_rate_day = len(admitted_emerg) / max_possible if max_possible > 0 else 1.0
            total_emerg_rate += emerg_rate_day

            # Paso 3: electivos (los del schedule de este individuo)
            day_elective_ids = scheduled_ids.get(day, [])
            # Reconstruye pacientes electivos del schedule
            elective_lookup = {p.patient_id: p for p in elective_waitlist}
            day_electives_admitted = []
            for pid in day_elective_ids:
                if pid in elective_lookup:
                    ep = elective_lookup[pid].clone()
                    bid = assign_bed(ep, occupied)
                    if bid is not None:
                        ep.assigned_bed  = bid
                        ep.admission_day = day
                        ep.delay = max(0, day - (ep.scheduled_day or day))
                        occupied.add(bid)
                        day_patients_count += 1
                        total_delay_score += ep.delay * ep.loss_of_chance
                        day_electives_admitted.append(ep)

            total_beds_used += day_patients_count

            # Avanzar estado
            next_active = []
            for p in still_active:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    next_active.append(p)
            for p in admitted_emerg:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    p.category = "current"
                    next_active.append(p)
            for p in day_electives_admitted:
                p.remaining_los -= 1
                if p.remaining_los > 0:
                    p.category = "current"
                    next_active.append(p)

            active_patients = next_active

        # Cálculo de métricas finales
        max_possible_beds = MAX_BEDS_ACTIVE * PLANNING_HORIZON
        f1_occ   = total_beds_used / max_possible_beds        # 0..1 (maximizar)
        f2_delay = total_delay_score                           # minimizar
        f3_emerg = total_emerg_rate / PLANNING_HORIZON        # 0..1 (maximizar)

        # Convertimos todo a minimización para NSGA-II
        self.fitness = (1.0 - f1_occ, f2_delay, 1.0 - f3_emerg)

    # ── Copia ────────────────────────────────
    def clone(self) -> "Individual":
        new_ind = Individual()
        new_ind.schedule = {
            day: [p.clone() for p in plist]
            for day, plist in self.schedule.items()
        }
        new_ind.fitness  = self.fitness
        new_ind.rank     = self.rank
        new_ind.crowding = self.crowding
        return new_ind

    def __repr__(self) -> str:
        return (f"Individual(rank={self.rank}, "
                f"f=({self.fitness[0]:.3f},{self.fitness[1]:.3f},{self.fitness[2]:.3f}))")
