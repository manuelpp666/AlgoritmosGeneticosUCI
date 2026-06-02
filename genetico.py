"""
genetico.py — Núcleo del Algoritmo Genético para asignación de camas UCI.

Cromosoma: vector de N tuplas (dia_ingreso, id_cama)
  - dia_ingreso ∈ {0,1,...,7}  (0 = no admitido esta semana)
  - id_cama    ∈ {1,...,28}    (irrelevante si dia=0)

Operadores implementados:
  - Inicialización aleatoria respetando especialidad
  - Cruce de dos puntos (Two-Point Crossover)
  - Mutación múltiple (día y/o cama)
  - Selección por torneo
  - Elitismo
"""

import random
import copy
from typing import Optional
from pacientes import Paciente
from config import (
    CAMAS, CAMAS_POR_TIPO, DIAS_SEMANA, DIA_NO_ADMITIDO,
    capacidad_dia,
    PENALIZACION_ESPECIALIDAD, PENALIZACION_SOLAPAMIENTO,
    PENALIZACION_MEDICOS, PENALIZACION_DIA_INVALIDO,
    TAMANIO_POBLACION, NUM_GENERACIONES,
    TASA_CRUCE, TASA_MUTACION, TORNEO_K, ELITISMO,
)

# ══════════════════════════════════════════════════════════════════
#  GEN y CROMOSOMA
# ══════════════════════════════════════════════════════════════════

# Gen = (dia_ingreso: int, id_cama: int)
# Cromosoma = list[Gen]  con longitud = número de pacientes

def _camas_compatibles(paciente: Paciente) -> list[int]:
    """Devuelve los IDs de camas compatibles con la especialidad del paciente."""
    return CAMAS_POR_TIPO.get(paciente.especialidad_requerida, [])


def gen_aleatorio(paciente: Paciente) -> tuple[int, int]:
    """Genera un gen (día, cama) válido para un paciente dado."""
    camas = _camas_compatibles(paciente)
    dia   = random.randint(1, DIAS_SEMANA)
    cama  = random.choice(camas)
    return (dia, cama)


def cromosoma_aleatorio(pacientes: list[Paciente]) -> list[tuple[int, int]]:
    """Genera un cromosoma aleatorio respetando la especialidad de cada paciente."""
    return [gen_aleatorio(p) for p in pacientes]


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN DE APTITUD (FITNESS)
# ══════════════════════════════════════════════════════════════════

def calcular_fitness(cromosoma: list[tuple[int, int]],
                     pacientes: list[Paciente]) -> float:
    """
    Calcula el fitness de un cromosoma.

    Retorna un escalar: mayor valor → mejor solución.

    Componentes:
      + Prioridad atendida  (gravedad × admitido)
      - Retraso de ingreso  (días de retraso × pérdida_oportunidad)
      - Penalizaciones por restricciones violadas
    """
    n = len(pacientes)
    fitness = 0.0
    penalizacion = 0.0

    # ── Índice: cama → lista de (dia_ingreso, dia_salida, idx_paciente)
    ocupacion_cama: dict[int, list[tuple[int, int, int]]] = {k: [] for k in CAMAS}

    # ── Índice: día → lista de pacientes admitidos ese día
    admitidos_por_dia: dict[int, list[int]] = {d: [] for d in range(1, DIAS_SEMANA + 1)}

    for i, (dia, cama) in enumerate(cromosoma):
        p = pacientes[i]

        # ── Validación de día ──────────────────────────────────────
        if dia == DIA_NO_ADMITIDO:
            # Paciente no admitido: no suma, no penaliza (sólo pierde la gravedad)
            continue

        if dia < 1 or dia > DIAS_SEMANA:
            penalizacion += PENALIZACION_DIA_INVALIDO
            continue

        # ── A) Restricción de especialidad ────────────────────────
        tipo_cama = CAMAS.get(cama)
        if tipo_cama != p.especialidad_requerida:
            penalizacion += PENALIZACION_ESPECIALIDAD
            # Aun así continuamos para evaluar otras restricciones
            # (no descartamos el gen completamente para no perder info de fitness)

        dia_salida = dia + p.los_estimado  # día en que libera la cama

        # Registramos la ocupación para verificar solapamientos
        ocupacion_cama[cama].append((dia, dia_salida, i))
        admitidos_por_dia[dia].append(i)

        # ── Puntuación positiva: prioridad atendida ───────────────
        fitness += p.gravedad * 10  # escala × 10 para magnitud manejable

        # ── Puntuación negativa: retraso ─────────────────────────
        retraso = max(0, dia - p.fecha_esperada)
        fitness -= retraso * p.gravedad * 2

    # ── B) Restricción de Recurso Dual (médicos por día) ──────────
    for dia in range(1, DIAS_SEMANA + 1):
        cap = capacidad_dia(dia)
        total_dia = len(admitidos_por_dia[dia])
        if total_dia > cap:
            exceso = total_dia - cap
            penalizacion += PENALIZACION_MEDICOS * exceso

    # ── C) Restricción de solapamiento y esterilización ───────────
    for cama_id, periodos in ocupacion_cama.items():
        if len(periodos) < 2:
            continue
        # Ordenar por día de ingreso
        periodos_ord = sorted(periodos, key=lambda x: x[0])
        for k in range(len(periodos_ord) - 1):
            _, salida_a, _ = periodos_ord[k]
            entrada_b, _, _ = periodos_ord[k + 1]
            # Día de esterilización: B debe entrar ESTRICTAMENTE después de salida_a
            # (salida_a ya incluye el día de alta → el siguiente puede entrar al día siguiente)
            if entrada_b <= salida_a:
                # Solapamiento o no se respeta la holgura de esterilización
                penalizacion += PENALIZACION_SOLAPAMIENTO

    return fitness - penalizacion


# ══════════════════════════════════════════════════════════════════
#  OPERADORES GENÉTICOS
# ══════════════════════════════════════════════════════════════════

def cruce_dos_puntos(padre1: list, padre2: list) -> tuple[list, list]:
    """
    Two-Point Crossover.
    Intercambia el segmento entre dos puntos de corte aleatorios.
    """
    n = len(padre1)
    if n < 3:
        return copy.deepcopy(padre1), copy.deepcopy(padre2)

    p1, p2 = sorted(random.sample(range(1, n), 2))
    hijo1 = padre1[:p1] + padre2[p1:p2] + padre1[p2:]
    hijo2 = padre2[:p1] + padre1[p1:p2] + padre2[p2:]
    return hijo1, hijo2


def mutar(cromosoma: list[tuple[int, int]],
          pacientes: list[Paciente],
          tasa: float = TASA_MUTACION) -> list[tuple[int, int]]:
    """
    Mutación gen a gen con probabilidad `tasa`.
    Tres tipos de mutación posibles:
      1. Cambiar día de ingreso (±1 o totalmente aleatorio)
      2. Cambiar cama (dentro de camas compatibles)
      3. Marcar paciente como no admitido (dia=0) – exploración de no-admisión
    """
    nuevo = list(cromosoma)
    for i, (dia, cama) in enumerate(nuevo):
        if random.random() < tasa:
            p = pacientes[i]
            tipo_mut = random.choices(
                ["dia", "cama", "no_admitir", "reparar"],
                weights=[0.40, 0.35, 0.10, 0.15],
                k=1
            )[0]

            if tipo_mut == "dia":
                if random.random() < 0.5 and 1 <= dia <= DIAS_SEMANA:
                    # Mutación pequeña: ±1
                    nuevo_dia = dia + random.choice([-1, 1])
                    nuevo_dia = max(1, min(DIAS_SEMANA, nuevo_dia))
                else:
                    nuevo_dia = random.randint(1, DIAS_SEMANA)
                nuevo[i] = (nuevo_dia, cama)

            elif tipo_mut == "cama":
                camas = _camas_compatibles(p)
                nueva_cama = random.choice(camas)
                nuevo[i] = (dia, nueva_cama)

            elif tipo_mut == "no_admitir":
                nuevo[i] = (DIA_NO_ADMITIDO, cama)

            elif tipo_mut == "reparar":
                # Reparación: fuerza compatibilidad de especialidad
                nuevo[i] = gen_aleatorio(p)

    return nuevo


# ══════════════════════════════════════════════════════════════════
#  SELECCIÓN POR TORNEO
# ══════════════════════════════════════════════════════════════════

def seleccion_torneo(poblacion: list[list],
                     fitness_vals: list[float],
                     k: int = TORNEO_K) -> list[tuple[int, int]]:
    """
    Selecciona un individuo mediante torneo de tamaño k.
    Devuelve una copia del ganador.
    """
    participantes = random.sample(range(len(poblacion)), k)
    ganador_idx   = max(participantes, key=lambda idx: fitness_vals[idx])
    return copy.deepcopy(poblacion[ganador_idx])


# ══════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def ejecutar_ag(pacientes: list[Paciente],
                callback_generacion=None) -> tuple[list, float, list[float]]:
    """
    Ejecuta el Algoritmo Genético completo.

    Args:
        pacientes:            Lista de pacientes en espera.
        callback_generacion:  Función opcional callback(gen, mejor_fitness, avg_fitness)
                              para monitoreo en tiempo real.

    Returns:
        (mejor_cromosoma, mejor_fitness, historial_fitness)
        historial_fitness: lista del mejor fitness por generación
    """
    n = len(pacientes)
    historial_mejor  = []
    historial_avg    = []

    # ── Inicialización ────────────────────────────────────────────
    poblacion = [cromosoma_aleatorio(pacientes) for _ in range(TAMANIO_POBLACION)]

    mejor_global       = None
    mejor_fitness_global = float("-inf")

    for gen in range(NUM_GENERACIONES):

        # ── Evaluación ────────────────────────────────────────────
        fitness_vals = [calcular_fitness(c, pacientes) for c in poblacion]

        # Mejor de esta generación
        idx_mejor = max(range(len(fitness_vals)), key=lambda i: fitness_vals[i])
        if fitness_vals[idx_mejor] > mejor_fitness_global:
            mejor_fitness_global = fitness_vals[idx_mejor]
            mejor_global         = copy.deepcopy(poblacion[idx_mejor])

        historial_mejor.append(mejor_fitness_global)
        historial_avg.append(sum(fitness_vals) / len(fitness_vals))

        if callback_generacion:
            callback_generacion(gen + 1, mejor_fitness_global, historial_avg[-1])

        # ── Elitismo: conservar los N mejores ─────────────────────
        pares_elite = sorted(
            zip(fitness_vals, poblacion),
            key=lambda x: x[0],
            reverse=True
        )
        nueva_poblacion = [copy.deepcopy(ind) for _, ind in pares_elite[:ELITISMO]]

        # ── Generar descendencia ──────────────────────────────────
        while len(nueva_poblacion) < TAMANIO_POBLACION:
            p1 = seleccion_torneo(poblacion, fitness_vals)
            p2 = seleccion_torneo(poblacion, fitness_vals)

            if random.random() < TASA_CRUCE:
                h1, h2 = cruce_dos_puntos(p1, p2)
            else:
                h1, h2 = copy.deepcopy(p1), copy.deepcopy(p2)

            h1 = mutar(h1, pacientes)
            h2 = mutar(h2, pacientes)

            nueva_poblacion.append(h1)
            if len(nueva_poblacion) < TAMANIO_POBLACION:
                nueva_poblacion.append(h2)

        poblacion = nueva_poblacion

    return mejor_global, mejor_fitness_global, historial_mejor
