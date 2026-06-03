"""
config.py - Configuración global del sistema UCI NSGA-II
Hospital Regional de Lambayeque
"""

# ──────────────────────────────────────────────
#  ESTRUCTURA DE CAMAS
# ──────────────────────────────────────────────
BEDS = {
    "adult":    list(range(1,  19)),   # camas  1-18  adulto
    "pediatric":list(range(19, 23)),   # camas 19-22  pediátrico
    "neonatal": list(range(23, 29)),   # camas 23-28  neonatal
}
TOTAL_BEDS = sum(len(v) for v in BEDS.values())   # 28

BED_TYPE_MAP: dict[int, str] = {}
for btype, ids in BEDS.items():
    for bid in ids:
        BED_TYPE_MAP[bid] = btype

# ──────────────────────────────────────────────
#  REGLA MÉDICO / PACIENTE  (1:6)
# ──────────────────────────────────────────────
DOCTORS_PER_SHIFT_OPTIONS = {5: 28, 4: 24, 3: 18}   # médicos → camas_max
DEFAULT_DOCTORS = 5
MAX_BEDS_ACTIVE = DOCTORS_PER_SHIFT_OPTIONS[DEFAULT_DOCTORS]

# ──────────────────────────────────────────────
#  PARÁMETROS DEL ALGORITMO GENÉTICO
# ──────────────────────────────────────────────
POPULATION_SIZE   = 100
MAX_GENERATIONS   = 200
CROSSOVER_RATE    = 0.9
MUTATION_RATE     = 0.2
PLANNING_HORIZON  = 7          # días de planificación

# ──────────────────────────────────────────────
#  POISSON: llegadas diarias de emergencias
# ──────────────────────────────────────────────
POISSON_LAMBDA = 3.0           # media de llegadas de emergencia por día

# ──────────────────────────────────────────────
#  LOSS OF CHANCE  (gravedad electivos)
# ──────────────────────────────────────────────
LOSS_OF_CHANCE_VALUES = [0.1, 0.5, 0.9]

# ──────────────────────────────────────────────
#  TIPO DE PACIENTE  (categorías UCI)
# ──────────────────────────────────────────────
PATIENT_TYPES = ["adult", "pediatric", "neonatal"]

# ──────────────────────────────────────────────
#  SEEDS  (reproducibilidad)
# ──────────────────────────────────────────────
RANDOM_SEED = 42
