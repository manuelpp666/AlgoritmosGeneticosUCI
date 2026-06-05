from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional
from config import TIPOS_PACIENTE, VALORES_PÉRDIDA_OPORTUNIDAD


@dataclass
class Paciente:
    id_paciente: str
    tipo_paciente: str
    categoría: str
    los: int
    los_restante: int
    día_programado: Optional[int] = None
    pérdida_oportunidad: float = 0.1
    día_ingreso: Optional[int] = None
    cama_asignada: Optional[int] = None
    retraso: int = 0

    def es_actual(self) -> bool:
        return self.categoría == "actual"

    def es_emergencia(self) -> bool:
        return self.categoría == "emergencia"

    def es_electivo(self) -> bool:
        return self.categoría == "electivo"

    def clonar(self) -> "Paciente":
        return Paciente(
            id_paciente=self.id_paciente,
            tipo_paciente=self.tipo_paciente,
            categoría=self.categoría,
            los=self.los,
            los_restante=self.los_restante,
            día_programado=self.día_programado,
            pérdida_oportunidad=self.pérdida_oportunidad,
            día_ingreso=self.día_ingreso,
            cama_asignada=self.cama_asignada,
            retraso=self.retraso,
        )


def generar_pacientes_actuales(rng: random.Random) -> list[Paciente]:
    pacientes = []
    id_pac = 10001

    for _ in range(12):
        los = rng.randint(2, 7)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente=str(id_pac),
            tipo_paciente="adulto",
            categoría="actual",
            los=los,
            los_restante=rem,
        ))
        id_pac += 1

    for _ in range(3):
        los = rng.randint(2, 6)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente=str(id_pac),
            tipo_paciente="pediátrico",
            categoría="actual",
            los=los,
            los_restante=rem,
        ))
        id_pac += 1

    for _ in range(4):
        los = rng.randint(3, 10)
        rem = rng.randint(1, los)
        pacientes.append(Paciente(
            id_paciente=str(id_pac),
            tipo_paciente="neonatal",
            categoría="actual",
            los=los,
            los_restante=rem,
        ))
        id_pac += 1

    return pacientes


def generar_lista_espera_electivos(rng: random.Random, n: int = 40) -> list[Paciente]:
    pacientes = []
    id_pac = 30001
    for i in range(n):
        tipo_pac = rng.choice(TIPOS_PACIENTE)
        los = rng.randint(1, 7)
        pérdida = rng.choice(VALORES_PÉRDIDA_OPORTUNIDAD)
        día_prog = rng.randint(1, 7)
        pacientes.append(Paciente(
            id_paciente=str(id_pac),
            tipo_paciente=tipo_pac,
            categoría="electivo",
            los=los,
            los_restante=los,
            día_programado=día_prog,
            pérdida_oportunidad=pérdida,
        ))
        id_pac += 1
    return pacientes


def generar_pacientes_emergencia(rng: random.Random, día: int, cantidad: int) -> list[Paciente]:
    pacientes = []
    id_base = 20000 + día * 100
    for i in range(cantidad):
        tipo_pac = rng.choice(TIPOS_PACIENTE)
        los = rng.randint(1, 5)
        pacientes.append(Paciente(
            id_paciente=str(id_base + i + 1),
            tipo_paciente=tipo_pac,
            categoría="emergencia",
            los=los,
            los_restante=los,
            día_ingreso=día,
        ))
    return pacientes
