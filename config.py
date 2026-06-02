"""
config.py — Parámetros globales del sistema UCI del Hospital Regional Lambayeque.
"""

# ─────────────────────────────────────────────
#  CAMAS UCI – Hospital Regional Lambayeque
# ─────────────────────────────────────────────
TOTAL_CAMAS = 28

# Camas 1-18: Adulto | 19-22: Pediátrica | 23-28: Neonatal
CAMAS = {}
for i in range(1, 19):
    CAMAS[i] = "Adulto"
for i in range(19, 23):
    CAMAS[i] = "Pediatrica"
for i in range(23, 29):
    CAMAS[i] = "Neonatal"

# Camas agrupadas por especialidad
CAMAS_POR_TIPO = {
    "Adulto":     [i for i, t in CAMAS.items() if t == "Adulto"],
    "Pediatrica": [i for i, t in CAMAS.items() if t == "Pediatrica"],
    "Neonatal":   [i for i, t in CAMAS.items() if t == "Neonatal"],
}

# ─────────────────────────────────────────────
#  RESTRICCIÓN DE RECURSO DUAL (médicos/turno)
# ─────────────────────────────────────────────
CAPACIDAD_POR_MEDICOS = {5: 28, 4: 24, 3: 18}

# Médicos disponibles por día (lun=1 … dom=7)
# Puedes modificar este diccionario según la realidad del hospital
MEDICOS_POR_DIA = {
    1: 5,  # lunes
    2: 5,  # martes
    3: 4,  # miércoles
    4: 5,  # jueves
    5: 5,  # viernes
    6: 3,  # sábado
    7: 3,  # domingo
}

def capacidad_dia(dia: int) -> int:
    """Devuelve la capacidad máxima de camas operativas para un día dado."""
    medicos = MEDICOS_POR_DIA.get(dia, 5)
    return CAPACIDAD_POR_MEDICOS.get(medicos, 28)

# ─────────────────────────────────────────────
#  PARÁMETROS DEL ALGORITMO GENÉTICO
# ─────────────────────────────────────────────
TAMANIO_POBLACION   = 120
NUM_GENERACIONES    = 150
TASA_CRUCE          = 0.85
TASA_MUTACION       = 0.15
TORNEO_K            = 5       # participantes en selección por torneo
ELITISMO            = 5       # individuos élite que pasan directamente

# ─────────────────────────────────────────────
#  PENALIZACIONES DE FITNESS
# ─────────────────────────────────────────────
PENALIZACION_ESPECIALIDAD  = 200   # cama incompatible con tipo de paciente
PENALIZACION_SOLAPAMIENTO  = 150   # camas con fechas superpuestas
PENALIZACION_MEDICOS       = 180   # excede límite de camas según médicos
PENALIZACION_DIA_INVALIDO  = 100   # día fuera del rango 0-7

# ─────────────────────────────────────────────
#  SEMANA DE PLANIFICACIÓN
# ─────────────────────────────────────────────
DIAS_SEMANA = 7    # Lunes=1 … Domingo=7
DIA_NO_ADMITIDO = 0
