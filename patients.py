"""
patients.py - Modelos de datos de pacientes y generación de datos de ejemplo
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from config import PATIENT_TYPES, LOSS_OF_CHANCE_VALUES, RANDOM_SEED


# ──────────────────────────────────────────────
#  MODELO DE PACIENTE
# ──────────────────────────────────────────────
@dataclass
class Patient:
    """Representa a un paciente de la UCI."""
    patient_id:   str
    patient_type: str          # 'adult' | 'pediatric' | 'neonatal'
    category:     str          # 'current' | 'emergency' | 'elective'
    los:          int          # Length of Stay total (días)
    remaining_los: int         # Días que faltan para el alta
    scheduled_day: Optional[int] = None   # Día programado (electivos)
    loss_of_chance: float = 0.1           # Gravedad (0.1, 0.5, 0.9)
    admission_day:  Optional[int] = None  # Día real de ingreso
    assigned_bed:   Optional[int] = None  # Cama asignada
    delay:          int = 0               # Días de retraso acumulado

    def is_current(self)   -> bool: return self.category == "current"
    def is_emergency(self) -> bool: return self.category == "emergency"
    def is_elective(self)  -> bool: return self.category == "elective"

    def clone(self) -> "Patient":
        return Patient(
            patient_id    = self.patient_id,
            patient_type  = self.patient_type,
            category      = self.category,
            los           = self.los,
            remaining_los = self.remaining_los,
            scheduled_day = self.scheduled_day,
            loss_of_chance= self.loss_of_chance,
            admission_day = self.admission_day,
            assigned_bed  = self.assigned_bed,
            delay         = self.delay,
        )


# ──────────────────────────────────────────────
#  GENERADORES DE DATOS DE EJEMPLO
# ──────────────────────────────────────────────
def generate_current_patients(rng: random.Random) -> list[Patient]:
    """
    Crea pacientes actualmente internados en la UCI.
    IDs: 10001, 10002, ...
    Distribución de camas: 18 adulto, 4 pediátrico, 6 neonatal.
    """
    patients = []
    pid = 10001

    # Adultos (12 de 18 camas ocupadas)
    for _ in range(12):
        los = rng.randint(2, 7)
        rem = rng.randint(1, los)
        patients.append(Patient(
            patient_id    = str(pid),
            patient_type  = "adult",
            category      = "current",
            los           = los,
            remaining_los = rem,
        ))
        pid += 1

    # Pediátricos (3 de 4 camas ocupadas)
    for _ in range(3):
        los = rng.randint(2, 6)
        rem = rng.randint(1, los)
        patients.append(Patient(
            patient_id    = str(pid),
            patient_type  = "pediatric",
            category      = "current",
            los           = los,
            remaining_los = rem,
        ))
        pid += 1

    # Neonatales (4 de 6 camas ocupadas)
    for _ in range(4):
        los = rng.randint(3, 10)
        rem = rng.randint(1, los)
        patients.append(Patient(
            patient_id    = str(pid),
            patient_type  = "neonatal",
            category      = "current",
            los           = los,
            remaining_los = rem,
        ))
        pid += 1

    return patients


def generate_elective_waitlist(rng: random.Random, n: int = 40) -> list[Patient]:
    """
    Lista de espera de pacientes electivos.
    IDs: 30001, 30002, ...
    """
    patients = []
    pid = 30001
    for i in range(n):
        ptype = rng.choice(PATIENT_TYPES)
        los   = rng.randint(1, 7)
        loc   = rng.choice(LOSS_OF_CHANCE_VALUES)
        sday  = rng.randint(1, 7)
        patients.append(Patient(
            patient_id    = str(pid),
            patient_type  = ptype,
            category      = "elective",
            los           = los,
            remaining_los = los,
            scheduled_day = sday,
            loss_of_chance= loc,
        ))
        pid += 1
    return patients


def generate_emergency_patients(rng: random.Random, day: int, count: int) -> list[Patient]:
    """
    Genera `count` pacientes de emergencia para un día dado.
    IDs: 20001, 20002, ...  (offset por día para unicidad)
    """
    patients = []
    base_pid = 20000 + day * 100
    for i in range(count):
        ptype = rng.choice(PATIENT_TYPES)
        los   = rng.randint(1, 5)
        patients.append(Patient(
            patient_id    = str(base_pid + i + 1),
            patient_type  = ptype,
            category      = "emergency",
            los           = los,
            remaining_los = los,
            admission_day = day,
        ))
    return patients
