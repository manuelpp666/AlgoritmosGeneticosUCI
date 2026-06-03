"""
patients.py - Modelos de datos de pacientes y generación de datos de ejemplo
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from config import TIPOS_PACIENTE, VALORES_PÉRDIDA_OPORTUNIDAD, SEMILLA_ALEATORIA


# ──────────────────────────────────────────────
#  MODELO DE PACIENTE
# ──────────────────────────────────────────────
@dataclass
class Paciente:
    """Representa a un paciente de la UCI."""
    id_paciente:   str
    tipo_paciente: str          # 'adulto' | 'pediátrico' | 'neonatal'
    categoría:     str          # 'actual' | 'emergencia' | 'electivo'
    los:          int          # Length of Stay total (días)
    los_restante: int         # Días que faltan para el alta
    día_programado: Optional[int] = None   # Día programado (electivos)
    pérdida_oportunidad: float = 0.1           # Gravedad (0.1, 0.5, 0.9)
    día_ingreso:  Optional[int] = None  # Día real de ingreso
    cama_asignada:   Optional[int] = None  # Cama asignada
    retraso:          int = 0               # Días de retraso acumulado

    def es_actual(self)   -> bool: return self.categoría == "actual"
    def es_emergencia(self) -> bool: return self.categoría == "emergencia"
    def es_electivo(self)  -> bool: return self.categoría == "electivo"

    def clonar(self) -> "Paciente":
        return Paciente(
            id_paciente    = self.id_paciente,
            tipo_paciente  = self.tipo_paciente,
            categoría      = self.categoría,
            los           = self.los,
            los_restante = self.los_restante,
            día_programado = self.día_programado,
            pérdida_oportunidad= self.pérdida_oportunidad,
            día_ingreso = self.día_ingreso,
            cama_asignada  = self.cama_asignada,
            retraso         = self.retraso,
        )


# ──────────────────────────────────────────────
#  GENERADORES DE DATOS DE EJEMPLO
# ──────────────────────────────────────────────
def generar_pacientes_actuales(rng: random.Random) -> list[Paciente]:
    """
    Crea pacientes actualmente internados en la UCI.
    IDs: 10001, 10002, ...
    Distribución de camas: 18 adulto, 4 pediátrico, 6 neonatal.
    """
    pacientes = []
    id_pac = 10001

    # Adultos (12 de 18 camas ocupadas)
    for _ in range(12):
        los = rng.randint(2, 7)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente    = str(id_pac),
            tipo_paciente  = "adulto",
            categoría      = "actual",
            los           = los,
            los_restante = rem,
        ))
        id_pac += 1

    # Pediátricos (3 de 4 camas ocupadas)
    for _ in range(3):
        los = rng.randint(2, 6)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente    = str(id_pac),
            tipo_paciente  = "pediátrico",
            categoría      = "actual",
            los           = los,
            los_restante = rem,
        ))
        id_pac += 1

    # Neonatales (4 de 6 camas ocupadas)
    for _ in range(4):
        los = rng.randint(3, 10)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente    = str(id_pac),
            tipo_paciente  = "neonatal",
            categoría      = "actual",
            los           = los,
            los_restante = rem,
        ))
        id_pac += 1

    return pacientes


def generar_lista_espera_electivos(rng: random.Random, n: int = 40) -> list[Paciente]:
    """
    Lista de espera de pacientes electivos.
    IDs: 30001, 30002, ...
    """
    pacientes = []
    id_pac = 30001
    for i in range(n):
        tipo_pac = rng.choice(TIPOS_PACIENTE)
        los   = rng.randint(1, 7)
        pérd_op   = rng.choice(VALORES_PÉRDIDA_OPORTUNIDAD)
        día_prog  = rng.randint(1, 7)
        pacientes.append(Paciente(
            id_paciente    = str(id_pac),
            tipo_paciente  = tipo_pac,
            categoría      = "electivo",
            los           = los,
            los_restante = los,
            día_programado = día_prog,
            pérdida_oportunidad= pérd_op,
        ))
        id_pac += 1
    return pacientes


def generar_pacientes_emergencia(rng: random.Random, día: int, cantidad: int) -> list[Paciente]:
    """
    Genera `cantidad` pacientes de emergencia para un día dado.
    IDs: 20001, 20002, ...  (offset por día para unicidad)
    """
    pacientes = []
    id_base = 20000 + día * 100
    for i in range(cantidad):
        tipo_pac = rng.choice(TIPOS_PACIENTE)
        los   = rng.randint(1, 5)
        pacientes.append(Paciente(
            id_paciente    = str(id_base + i + 1),
            tipo_paciente  = tipo_pac,
            categoría      = "emergencia",
            los           = los,
            los_restante = los,
            día_ingreso = día,
        ))
    return pacientes
