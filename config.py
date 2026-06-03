"""
config.py - Configuración global del sistema UCI NSGA-II
Hospital Regional de Lambayeque
"""

# ──────────────────────────────────────────────
#  ESTRUCTURA DE CAMAS
# ──────────────────────────────────────────────
CAMAS = {
    "adulto":    list(range(1,  19)),   # camas  1-18  adulto
    "pediátrico":list(range(19, 23)),   # camas 19-22  pediátrico
    "neonatal": list(range(23, 29)),   # camas 23-28  neonatal
}
TOTAL_CAMAS = sum(len(v) for v in CAMAS.values())   # 28

MAPA_TIPO_CAMA: dict[int, str] = {}
for tipo_cama, ids in CAMAS.items():
    for id_cama in ids:
        MAPA_TIPO_CAMA[id_cama] = tipo_cama

# ──────────────────────────────────────────────
#  REGLA MÉDICO / PACIENTE  (1:6)
# ──────────────────────────────────────────────
OPCIONES_MÉDICOS_TURNO = {5: 28, 4: 24, 3: 18}   # médicos → camas_max
MÉDICOS_DEFECTO = 5
CAMAS_ACTIVAS_MAX = OPCIONES_MÉDICOS_TURNO[MÉDICOS_DEFECTO]

# ──────────────────────────────────────────────
#  PARÁMETROS DEL ALGORITMO GENÉTICO
# ──────────────────────────────────────────────
TAMAÑO_POBLACIÓN   = 100
GENERACIONES_MAX   = 200
TASA_CRUZAMIENTO    = 0.9
TASA_MUTACIÓN     = 0.2
HORIZONTE_PLANIFICACIÓN  = 7          # días de planificación

# ──────────────────────────────────────────────
#  POISSON: llegadas diarias de emergencias
# ──────────────────────────────────────────────
LAMBDA_POISSON = 3.0           # media de llegadas de emergencia por día

# ──────────────────────────────────────────────
#  PÉRDIDA DE OPORTUNIDAD  (gravedad electivos)
# ──────────────────────────────────────────────
VALORES_PÉRDIDA_OPORTUNIDAD = [0.1, 0.5, 0.9]

# ──────────────────────────────────────────────
#  TIPO DE PACIENTE  (categorías UCI)
# ──────────────────────────────────────────────
TIPOS_PACIENTE = ["adulto", "pediátrico", "neonatal"]

# ──────────────────────────────────────────────
#  SEMILLA  (reproducibilidad)
# ──────────────────────────────────────────────
SEMILLA_ALEATORIA = 42
