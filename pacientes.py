"""
pacientes.py — Generación de la lista de espera de pacientes simulados.

Categorías clínicas (datos epidemiológicos Región Lambayeque):
  - Quirúrgico Complejo  (Neurocirugía/Trauma):  5-10 días UCI
  - Médico Agudo / Sepsis severa:                4-7  días UCI
  - Post-operatorio alta complejidad:            2-4  días UCI
"""

import random
from dataclasses import dataclass, field
from typing import Literal

# Tipos de especialidad de paciente (deben coincidir con los tipos de cama)
TipoEspecialidad = Literal["Adulto", "Pediatrica", "Neonatal"]

# Categorías clínicas
CATEGORIAS = [
    "Quirurgico_Complejo",
    "Medico_Agudo_Sepsis",
    "Postoperatorio_Alta_Complejidad",
]

# LOS (Length of Stay) en días por categoría  →  (min, max)
LOS_POR_CATEGORIA = {
    "Quirurgico_Complejo":              (5, 10),
    "Medico_Agudo_Sepsis":              (4,  7),
    "Postoperatorio_Alta_Complejidad":  (2,  4),
}

# Gravedad base por categoría (0-1, inspirada en el Artículo 1 / pérdida de oportunidad)
GRAVEDAD_BASE = {
    "Quirurgico_Complejo":             0.90,
    "Medico_Agudo_Sepsis":             0.75,
    "Postoperatorio_Alta_Complejidad": 0.50,
}

# Distribución de especialidad de cama requerida por categoría
DISTRIBUCION_ESPECIALIDAD = {
    "Quirurgico_Complejo":             {"Adulto": 0.70, "Pediatrica": 0.20, "Neonatal": 0.10},
    "Medico_Agudo_Sepsis":             {"Adulto": 0.60, "Pediatrica": 0.15, "Neonatal": 0.25},
    "Postoperatorio_Alta_Complejidad": {"Adulto": 0.65, "Pediatrica": 0.20, "Neonatal": 0.15},
}


@dataclass
class Paciente:
    id: int
    nombre: str
    categoria: str
    especialidad_requerida: TipoEspecialidad
    gravedad: float          # 0.0 – 1.0 (pérdida de oportunidad)
    los_estimado: int        # días de estancia estimados
    fecha_esperada: int      # día ideal de ingreso (1-7); 0 = ingreso urgente


def _elegir_especialidad(categoria: str) -> TipoEspecialidad:
    dist = DISTRIBUCION_ESPECIALIDAD[categoria]
    opciones = list(dist.keys())
    pesos   = list(dist.values())
    return random.choices(opciones, weights=pesos, k=1)[0]


def _jitter_gravedad(base: float, sigma: float = 0.08) -> float:
    """Agrega variabilidad aleatoria a la gravedad base."""
    valor = base + random.gauss(0, sigma)
    return round(max(0.05, min(0.99, valor)), 3)


def generar_lista_espera(n_pacientes: int = 40, semilla: int = None) -> list[Paciente]:
    """
    Genera N pacientes simulados para la lista de espera semanal UCI.

    Args:
        n_pacientes: Total de pacientes en lista de espera.
        semilla: Semilla aleatoria para reproducibilidad.

    Returns:
        Lista de objetos Paciente.
    """
    if semilla is not None:
        random.seed(semilla)

    pacientes = []
    nombres_base = [
        "García", "López", "Martínez", "Rodríguez", "Torres",
        "Ramírez", "Flores", "Mendoza", "Quispe", "Huanca",
        "Castro", "Díaz", "Vargas", "Reyes", "Morales",
        "Sánchez", "Jiménez", "Pérez", "Gómez", "Herrera",
        "Núñez", "Salinas", "Vera", "Paredes", "Chávez",
        "Cabrera", "Rojas", "Espinoza", "Aguilar", "Medina",
        "Muñoz", "Vega", "Ortega", "Castillo", "Ramos",
        "Gutiérrez", "Ríos", "Cruz", "Delgado", "Guerrero",
    ]

    # Distribución de categorías aproximada a la epidemiología regional
    pesos_categoria = [0.35, 0.40, 0.25]

    for i in range(n_pacientes):
        categoria = random.choices(CATEGORIAS, weights=pesos_categoria, k=1)[0]
        los_min, los_max = LOS_POR_CATEGORIA[categoria]
        especialidad   = _elegir_especialidad(categoria)
        gravedad       = _jitter_gravedad(GRAVEDAD_BASE[categoria])
        los            = random.randint(los_min, los_max)
        fecha_esperada = random.randint(1, 7)  # día ideal de ingreso en la semana
        nombre = nombres_base[i % len(nombres_base)] + f"-{i+1:03d}"

        pacientes.append(Paciente(
            id=i,
            nombre=nombre,
            categoria=categoria,
            especialidad_requerida=especialidad,
            gravedad=gravedad,
            los_estimado=los,
            fecha_esperada=fecha_esperada,
        ))

    # Ordenar por gravedad descendente (más graves primero en la lista)
    pacientes.sort(key=lambda p: p.gravedad, reverse=True)

    return pacientes


def mostrar_resumen(pacientes: list[Paciente]) -> None:
    """Imprime un resumen estadístico de la lista de espera."""
    from tabulate import tabulate

    filas = []
    for p in pacientes:
        filas.append([
            p.id,
            p.nombre,
            p.categoria.replace("_", " "),
            p.especialidad_requerida,
            f"{p.gravedad:.3f}",
            p.los_estimado,
            p.fecha_esperada,
        ])

    headers = ["ID", "Paciente", "Categoría", "Especialidad", "Gravedad", "LOS(días)", "Día Ideal"]
    print("\n" + "=" * 90)
    print(f"  LISTA DE ESPERA UCI – Hospital Regional Lambayeque  ({len(pacientes)} pacientes)")
    print("=" * 90)
    print(tabulate(filas, headers=headers, tablefmt="rounded_grid"))

    # Estadísticas
    graves = sum(1 for p in pacientes if p.gravedad >= 0.80)
    print(f"\n  Graves (≥0.80): {graves} | "
          f"Adulto: {sum(1 for p in pacientes if p.especialidad_requerida=='Adulto')} | "
          f"Pediátrica: {sum(1 for p in pacientes if p.especialidad_requerida=='Pediatrica')} | "
          f"Neonatal: {sum(1 for p in pacientes if p.especialidad_requerida=='Neonatal')}\n")
