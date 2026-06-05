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
    CAMAS, MAPA_TIPO_CAMA, CAMAS_ACTIVAS_MAX, TOTAL_CAMAS,
    HORIZONTE_PLANIFICACIÓN, LAMBDA_POISSON,
)
from patients import Paciente, generar_pacientes_emergencia


# ──────────────────────────────────────────────
#  UTILIDADES
# ──────────────────────────────────────────────
def muestra_poisson(lambda_par: float, rng: random.Random) -> int:
    """Generación de muestra Poisson (Knuth)."""
    L = math.exp(-lambda_par)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def camas_por_tipo(tipo_paciente: str) -> list[int]:
    """Devuelve lista de IDs de cama compatibles con el tipo de paciente."""
    return list(CAMAS[tipo_paciente])


def asignar_cama(paciente: Paciente, ocupadas: set[int]) -> Optional[int]:
    """
    Intenta asignar la primera cama libre del tipo correcto.
    Retorna el ID de cama o None si no hay disponible.
    """
    for id_cama in camas_por_tipo(paciente.tipo_paciente):
        if id_cama not in ocupadas and id_cama <= CAMAS_ACTIVAS_MAX:
            return id_cama
    return None


# ──────────────────────────────────────────────
#  CROMOSOMA
# ──────────────────────────────────────────────
class Individuo:
    """
    Un individuo = plan semanal de 7 días para la UCI.

    Atributos:
        programa  : dict[día → list[Paciente]]   plan electivos por día
        aptitud   : (f1, f2, f3) calculado tras evaluar()
        rango      : frente de Pareto
        hacinamiento  : distancia de hacinamiento NSGA-II
    """

    def __init__(self):
        self.programa: dict[int, list[Paciente]] = {d: [] for d in range(1, HORIZONTE_PLANIFICACIÓN + 1)}
        self.aptitud:  tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.rango:     int   = 0
        self.hacinamiento: float = 0.0

    # ── Construcción ──────────────────────────
    @classmethod
    def crear_aleatorio(
        cls,
        pacientes_actuales: list[Paciente],
        lista_espera_electivos: list[Paciente],
        rng: random.Random,
    ) -> "Individuo":
        """
        Genera un individuo aleatorio respetando la jerarquía de asignación.
        """
        ind = cls()
        grupo_electivos = [p.clonar() for p in lista_espera_electivos]
        rng.shuffle(grupo_electivos)

        # Puntero a los electivos aún no asignados
        índice_electivo = 0

        # Estado de la UCI: pacientes activos que "arrastran" al día siguiente
        pacientes_activos: list[Paciente] = [p.clonar() for p in pacientes_actuales]

        for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
            ocupadas: set[int] = set()

            # Paso 1: pacientes actuales (inamovibles)
            aún_activos: list[Paciente] = []
            for p in pacientes_activos:
                if p.los_restante > 0:
                    id_cama = asignar_cama(p, ocupadas)
                    if id_cama is not None:
                        p.cama_asignada = id_cama
                        ocupadas.add(id_cama)
                        aún_activos.append(p)
                    # Si no hay cama del tipo correcto disponible, caso excepcional ignorado

            # Paso 2: emergencias (Poisson)
            n_emergencias = muestra_poisson(LAMBDA_POISSON, rng)
            emergencias = generar_pacientes_emergencia(rng, día, n_emergencias)
            emergencias_admitidas: list[Paciente] = []
            for ep in emergencias:
                id_cama = asignar_cama(ep, ocupadas)
                if id_cama is not None:
                    ep.cama_asignada = id_cama
                    ep.día_ingreso = día
                    ocupadas.add(id_cama)
                    emergencias_admitidas.append(ep)

            # Paso 3: electivos (rellenar huecos)
            electivos_día: list[Paciente] = []
            while índice_electivo < len(grupo_electivos):
                ep = grupo_electivos[índice_electivo]
                id_cama = asignar_cama(ep, ocupadas)
                if id_cama is not None:
                    ep_copia = ep.clonar()
                    ep_copia.cama_asignada  = id_cama
                    ep_copia.día_ingreso = día
                    ep_copia.retraso = max(0, día - (ep.día_programado or día))
                    ocupadas.add(id_cama)
                    electivos_día.append(ep_copia)
                    índice_electivo += 1
                else:
                    break  # No quedan camas disponibles

            ind.programa[día] = electivos_día

            # Actualizar estado para el día siguiente
            próximos_activos: list[Paciente] = []
            for p in aún_activos:
                p.los_restante -= 1
                if p.los_restante > 0:
                    próximos_activos.append(p)
            for p in emergencias_admitidas:
                p.los_restante -= 1
                if p.los_restante > 0:
                    p.categoría = "actual"
                    próximos_activos.append(p)
            for p in electivos_día:
                p.los_restante -= 1
                if p.los_restante > 0:
                    p.categoría = "actual"
                    próximos_activos.append(p)

            pacientes_activos = próximos_activos

        return ind

    # ── Evaluación (fitness) ──────────────────
    def evaluar(
        self,
        pacientes_actuales: list[Paciente],
        lista_espera_electivos: list[Paciente],
        rng: random.Random,
    ) -> None:
        """
        Simula la semana completa y calcula (f1, f2, f3).
        f1 = tasa ocupación  (maximizar → convertida en minimización: 1-f1)
        f2 = índice retraso  (minimizar)
        f3 = tasa admisión emergencias (maximizar → 1-f3)
        """
        camas_totales_usadas   = 0
        puntuación_retraso_total = 0.0
        tasa_emergencias_total  = 0.0
        total_emergencias_máximo_posible = 0

        pacientes_activos: list[Paciente] = [p.clonar() for p in pacientes_actuales]

        # Reconstituimos un pool de electivos usando el programa
        ids_programados = {
            día: [p.id_paciente for p in plist]
            for día, plist in self.programa.items()
        }

        for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
            ocupadas: set[int] = set()
            cuenta_pacientes_día = 0

            # Paso 1: actuales
            aún_activos = []
            for p in pacientes_activos:
                if p.los_restante > 0:
                    id_cama = asignar_cama(p, ocupadas)
                    if id_cama is not None:
                        ocupadas.add(id_cama)
                        cuenta_pacientes_día += 1
                        aún_activos.append(p)

            # Paso 2: emergencias
            n_emergencias = muestra_poisson(LAMBDA_POISSON, rng)
            emergencias = generar_pacientes_emergencia(rng, día, n_emergencias)
            # Máximo posible según Poisson (usamos techo de 3*lambda como límite)
            máximo_posible = min(int(3 * LAMBDA_POISSON), CAMAS_ACTIVAS_MAX)
            total_emergencias_máximo_posible += máximo_posible

            emergencias_admitidas = []
            for ep in emergencias:
                id_cama = asignar_cama(ep, ocupadas)
                if id_cama is not None:
                    ep.cama_asignada = id_cama
                    ep.día_ingreso = día
                    ocupadas.add(id_cama)
                    cuenta_pacientes_día += 1
                    emergencias_admitidas.append(ep)

            tasa_emergencias_día = len(emergencias_admitidas) / máximo_posible if máximo_posible > 0 else 1.0
            tasa_emergencias_total += tasa_emergencias_día

            # Paso 3: electivos (los del programa de este individuo)
            ids_electivos_día = ids_programados.get(día, [])
            # Reconstruye pacientes electivos del programa
            búsqueda_electivos = {p.id_paciente: p for p in lista_espera_electivos}
            electivos_día_admitidos = []
            for id_pac in ids_electivos_día:
                if id_pac in búsqueda_electivos:
                    ep = búsqueda_electivos[id_pac].clonar()
                    id_cama = asignar_cama(ep, ocupadas)
                    if id_cama is not None:
                        ep.cama_asignada  = id_cama
                        ep.día_ingreso = día
                        ep.retraso = max(0, día - (ep.día_programado or día))
                        ocupadas.add(id_cama)
                        cuenta_pacientes_día += 1
                        puntuación_retraso_total += ep.retraso * ep.pérdida_oportunidad
                        electivos_día_admitidos.append(ep)

            camas_totales_usadas += cuenta_pacientes_día

            # Avanzar estado
            próximos_activos = []
            for p in aún_activos:
                p.los_restante -= 1
                if p.los_restante > 0:
                    próximos_activos.append(p)
            for p in emergencias_admitidas:
                p.los_restante -= 1
                if p.los_restante > 0:
                    p.categoría = "actual"
                    próximos_activos.append(p)
            for p in electivos_día_admitidos:
                p.los_restante -= 1
                if p.los_restante > 0:
                    p.categoría = "actual"
                    próximos_activos.append(p)

            pacientes_activos = próximos_activos

        # Cálculo de métricas finales
        camas_máximo_posible = CAMAS_ACTIVAS_MAX * HORIZONTE_PLANIFICACIÓN
        f1_ocup   = camas_totales_usadas / camas_máximo_posible        # 0..1 (maximizar)
        f2_retraso = puntuación_retraso_total                           # minimizar
        f3_emerg = tasa_emergencias_total / HORIZONTE_PLANIFICACIÓN        # 0..1 (maximizar)

        # Convertimos todo a minimización para NSGA-II
        self.aptitud = (1.0 - f1_ocup, f2_retraso, 1.0 - f3_emerg)

    # ── Copia ────────────────────────────────
    def clonar(self) -> "Individuo":
        nuevo_ind = Individuo()
        nuevo_ind.programa = {
            día: [p.clonar() for p in plist]
            for día, plist in self.programa.items()
        }
        nuevo_ind.aptitud  = self.aptitud
        nuevo_ind.rango     = self.rango
        nuevo_ind.hacinamiento = self.hacinamiento
        return nuevo_ind

    def __repr__(self) -> str:
        return (f"Individuo(rango={self.rango}, "
                f"apt=({self.aptitud[0]:.3f},{self.aptitud[1]:.3f},{self.aptitud[2]:.3f}))")
