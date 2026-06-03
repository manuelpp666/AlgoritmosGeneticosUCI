"""
nsga2.py - Núcleo del algoritmo NSGA-II
Non-dominated sorting, crowding distance, selection, crossover, mutación
Hospital Regional de Lambayeque
"""
from __future__ import annotations
import random
from typing import Callable

from config import (
    CAMAS_ACTIVAS_MAX, HORIZONTE_PLANIFICACIÓN,
    TASA_CRUZAMIENTO, TASA_MUTACIÓN,
    CAMAS,
)
from patients import Paciente
from chromosome import Individuo, asignar_cama, camas_por_tipo


# ══════════════════════════════════════════════
#  ORDENAMIENTO NO DOMINADO
# ══════════════════════════════════════════════
def domina(a: Individuo, b: Individuo) -> bool:
    """
    a domina a b si a es al menos tan bueno en todos los objetivos
    y estrictamente mejor en al menos uno. (minimización)
    """
    fa, fb = a.aptitud, b.aptitud
    al_menos_igual = all(fa[i] <= fb[i] for i in range(3))
    estrictamente_mejor = any(fa[i] <  fb[i] for i in range(3))
    return al_menos_igual and estrictamente_mejor


def ordenamiento_no_dominado_rápido(población: list[Individuo]) -> list[list[Individuo]]:
    """
    Algoritmo de ordenamiento no-dominado O(MN²).
    Retorna lista de frentes de Pareto [F1, F2, ...].
    """
    n = len(población)
    dominado_por: list[list[int]] = [[] for _ in range(n)]  # S_p
    contar_domina:    list[int]       = [0]  * n                 # n_p
    frentes: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if domina(población[i], población[j]):
                dominado_por[i].append(j)
            elif domina(población[j], población[i]):
                contar_domina[i] += 1
        if contar_domina[i] == 0:
            población[i].rango = 1
            frentes[0].append(i)

    frente_actual = 0
    while frentes[frente_actual]:
        siguiente_frente: list[int] = []
        for i in frentes[frente_actual]:
            for j in dominado_por[i]:
                contar_domina[j] -= 1
                if contar_domina[j] == 0:
                    población[j].rango = frente_actual + 2
                    siguiente_frente.append(j)
        frentes.append(siguiente_frente)
        frente_actual += 1

    # Convertir índices a individuos
    resultado = []
    for f in frentes:
        if f:
            resultado.append([población[i] for i in f])
    return resultado


# ══════════════════════════════════════════════
#  DISTANCIA DE HACINAMIENTO
# ══════════════════════════════════════════════
def asignación_distancia_hacinamiento(frente: list[Individuo]) -> None:
    """Asigna distancia de hacinamiento a cada individuo del frente."""
    n = len(frente)
    if n == 0:
        return
    for ind in frente:
        ind.hacinamiento = 0.0

    n_obj = 3
    for m in range(n_obj):
        frente.sort(key=lambda x: x.aptitud[m])
        frente[0].hacinamiento   = float("inf")
        frente[-1].hacinamiento  = float("inf")
        f_mín = frente[0].aptitud[m]
        f_máx = frente[-1].aptitud[m]
        denom = f_máx - f_mín if f_máx != f_mín else 1e-10
        for i in range(1, n - 1):
            frente[i].hacinamiento += (
                (frente[i + 1].aptitud[m] - frente[i - 1].aptitud[m]) / denom
            )


# ══════════════════════════════════════════════
#  SELECCIÓN BINARIA POR TORNEO
# ══════════════════════════════════════════════
def torneo_hacinado(a: Individuo, b: Individuo) -> Individuo:
    """Selección por torneo usando rango y hacinamiento (NSGA-II)."""
    if a.rango < b.rango:
        return a
    if b.rango < a.rango:
        return b
    return a if a.hacinamiento >= b.hacinamiento else b


def selección_torneo(
    población: list[Individuo],
    rng: random.Random,
) -> Individuo:
    """Selección por torneo binario."""
    i, j = rng.sample(range(len(población)), 2)
    return torneo_hacinado(población[i], población[j])


# ══════════════════════════════════════════════
#  CRUZAMIENTO
# ══════════════════════════════════════════════
def _contar_camas_día(
    día: int,
    electivos_día: list[Paciente],
    arrastrados: dict[int, list[Paciente]],
) -> int:
    """
    Cuenta camas ocupadas para un día dado considerando los pacientes
    arrastrados + los electivos programados ese día.
    """
    return len(arrastrados.get(día, [])) + len(electivos_día)


def cruzamiento(
    progenitor1: Individuo,
    progenitor2: Individuo,
    pacientes_actuales: list[Paciente],
    rng: random.Random,
) -> tuple[Individuo, Individuo]:
    """
    Cruzamiento de fechas de admisión entre dos pacientes electivos.
    Intercambia el día de ingreso de dos electivos aleatorios (uno de cada progenitor)
    y verifica que no se supere CAMAS_ACTIVAS_MAX.
    """
    if rng.random() > TASA_CRUZAMIENTO:
        return progenitor1.clonar(), progenitor2.clonar()

    hijo1 = progenitor1.clonar()
    hijo2 = progenitor2.clonar()

    # Recopilar todos los electivos programados en cada hijo
    def obtener_todos_electivos(ind: Individuo) -> list[tuple[int, int]]:
        """Retorna lista de (día, índice_en_programa_día) para cada electivo."""
        resultado = []
        for día, plist in ind.programa.items():
            for índice in range(len(plist)):
                resultado.append((día, índice))
        return resultado

    elec1 = obtener_todos_electivos(hijo1)
    elec2 = obtener_todos_electivos(hijo2)

    if not elec1 or not elec2:
        return hijo1, hijo2

    # Elegir un electivo de cada hijo
    d1, i1 = rng.choice(elec1)
    d2, i2 = rng.choice(elec2)

    p1 = hijo1.programa[d1][i1].clonar()
    p2 = hijo2.programa[d2][i2].clonar()

    # Intentar intercambio en hijo1: reemplazar p1 (en d1) por p2 (en d2)
    if _es_intercambio_viable(hijo1, d1, i1, p2, d2):
        hijo1.programa[d1][i1] = p2.clonar()
        hijo1.programa[d1][i1].día_ingreso = d1

    # Intentar intercambio en hijo2: reemplazar p2 (en d2) por p1 (en d1)
    if _es_intercambio_viable(hijo2, d2, i2, p1, d1):
        hijo2.programa[d2][i2] = p1.clonar()
        hijo2.programa[d2][i2].día_ingreso = d2

    return hijo1, hijo2


def _es_intercambio_viable(
    ind: Individuo,
    día: int,
    índice: int,
    nuevo_paciente: Paciente,
    día_original: int,
) -> bool:
    """
    Verifica que al insertar nuevo_paciente en `día` no se supere CAMAS_ACTIVAS_MAX.
    También verifica compatibilidad de tipo de cama.
    """
    # Comprobar que hay cama del tipo correcto disponible en ese día
    tipos_ocupados = {p.tipo_paciente for p in ind.programa[día]}
    disponibles = sum(
        1 for id_cama in camas_por_tipo(nuevo_paciente.tipo_paciente)
        if id_cama <= CAMAS_ACTIVAS_MAX
    )
    usadas_del_tipo = sum(
        1 for p in ind.programa[día]
        if p.tipo_paciente == nuevo_paciente.tipo_paciente
    )
    if usadas_del_tipo >= disponibles:
        return False

    # Verificar límite total del día
    total_día = sum(len(v) for v in ind.programa.values() if v)
    # Aproximación simple: si el día ya tiene muchos pacientes, rechazar
    if len(ind.programa[día]) >= CAMAS_ACTIVAS_MAX:
        return False

    return True


# ══════════════════════════════════════════════
#  MUTACIÓN
# ══════════════════════════════════════════════
def mutación(
    individuo: Individuo,
    lista_espera_electivos: list[Paciente],
    rng: random.Random,
) -> Individuo:
    """
    Mutación: elimina un electivo programado e intenta insertar otro
    de la lista de espera con LOS ≤ al eliminado.
    """
    if rng.random() > TASA_MUTACIÓN:
        return individuo

    mutante = individuo.clonar()

    # Recopilar electivos programados
    programados = [
        (día, idx)
        for día, plist in mutante.programa.items()
        for idx in range(len(plist))
    ]
    if not programados:
        return mutante

    # Elegir un electivo al azar para eliminar
    día, idx = rng.choice(programados)
    eliminado = mutante.programa[día].pop(idx)

    # Buscar candidato de reemplazo con LOS ≤ eliminado y tipo igual
    # (para respetar las camas por tipo)
    candidatos = [
        p for p in lista_espera_electivos
        if p.los <= eliminado.los
        and p.tipo_paciente == eliminado.tipo_paciente
        and not _ya_programado(p.id_paciente, mutante)
    ]

    if candidatos:
        # Prioriza mayor pérdida_oportunidad (urgencia)
        candidatos.sort(key=lambda p: -p.pérdida_oportunidad)
        nuevo_p = candidatos[0].clonar()
        nuevo_p.día_ingreso = día
        nuevo_p.retraso = max(0, día - (nuevo_p.día_programado or día))
        mutante.programa[día].append(nuevo_p)

    return mutante


def _ya_programado(id_pac: str, ind: Individuo) -> bool:
    for plist in ind.programa.values():
        for p in plist:
            if p.id_paciente == id_pac:
                return True
    return False


# ══════════════════════════════════════════════
#  REINSERCIÓN
# ══════════════════════════════════════════════
def reinserción(
    individuo: Individuo,
    lista_espera_electivos: list[Paciente],
    pacientes_actuales: list[Paciente],
    rng: random.Random,
) -> Individuo:
    """
    Si quedan camas vacías en algún día, intenta insertar electivos
    priorizando menor LOS y mayor pérdida_oportunidad.
    """
    ind = individuo.clonar()

    # Calcular cuántas camas hay disponibles por tipo por día (aproximación)
    for día in range(1, HORIZONTE_PLANIFICACIÓN + 1):
        # Contar camas usadas por tipo en este día
        usadas_por_tipo: dict[str, int] = {"adulto": 0, "pediátrico": 0, "neonatal": 0}
        for p in ind.programa[día]:
            usadas_por_tipo[p.tipo_paciente] += 1
        # También contar actuales que arrastran (simplificación)
        for tipo_pac, capacidad in [("adulto", 18), ("pediátrico", 4), ("neonatal", 6)]:
            máximo_tipo = min(capacidad, CAMAS_ACTIVAS_MAX)  # cap por tipo
            disponibles = máximo_tipo - usadas_por_tipo[tipo_pac]
            if disponibles <= 0:
                continue

            # Candidatos no programados del tipo correcto
            candidatos = sorted(
                [
                    p for p in lista_espera_electivos
                    if p.tipo_paciente == tipo_pac
                    and not _ya_programado(p.id_paciente, ind)
                ],
                key=lambda p: (p.los, -p.pérdida_oportunidad),
            )
            insertados = 0
            for p in candidatos:
                if insertados >= disponibles:
                    break
                nuevo_p = p.clonar()
                nuevo_p.día_ingreso = día
                nuevo_p.retraso = max(0, día - (nuevo_p.día_programado or día))
                ind.programa[día].append(nuevo_p)
                insertados += 1

    return ind


# ══════════════════════════════════════════════
#  SELECCIÓN DE PRÓXIMA GENERACIÓN
# ══════════════════════════════════════════════
def seleccionar_próxima_generación(
    combinado: list[Individuo],
    tamaño_pop: int,
) -> list[Individuo]:
    """
    Merge + NSGA-II selection: llena la nueva población con los mejores frentes
    usando distancia de hacinamiento como desempate en el frente límite.
    """
    frentes = ordenamiento_no_dominado_rápido(combinado)
    nueva_pop: list[Individuo] = []

    for frente in frentes:
        asignación_distancia_hacinamiento(frente)
        if len(nueva_pop) + len(frente) <= tamaño_pop:
            nueva_pop.extend(frente)
        else:
            # Tomar los mejores por distancia de hacinamiento hasta completar tamaño_pop
            faltantes = tamaño_pop - len(nueva_pop)
            frente.sort(key=lambda x: -x.hacinamiento)
            nueva_pop.extend(frente[:faltantes])
            break

    return nueva_pop
