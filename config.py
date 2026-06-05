CAMAS = {
    "adulto": list(range(1, 19)),
    "pediátrico": list(range(19, 23)),
    "neonatal": list(range(23, 29)),
}
TOTAL_CAMAS = sum(len(v) for v in CAMAS.values())

MAPA_TIPO_CAMA: dict[int, str] = {}
for tipo_cama, ids in CAMAS.items():
    for id_cama in ids:
        MAPA_TIPO_CAMA[id_cama] = tipo_cama

OPCIONES_MÉDICOS_TURNO = {5: 28, 4: 24, 3: 18}
MÉDICOS_DEFECTO = 4
CAMAS_ACTIVAS_MAX = OPCIONES_MÉDICOS_TURNO[MÉDICOS_DEFECTO]

TAMAÑO_POBLACIÓN = 100
GENERACIONES_MAX = 200
TASA_CRUZAMIENTO = 0.9
TASA_MUTACIÓN = 0.2
HORIZONTE_PLANIFICACIÓN = 7

LAMBDA_POISSON = 3.0

VALORES_PÉRDIDA_OPORTUNIDAD = [0.1, 0.5, 0.9]

TIPOS_PACIENTE = ["adulto", "pediátrico", "neonatal"]

SEMILLA_ALEATORIA = 42
