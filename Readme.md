| Archivo | Descripción |
| :--- | :--- |
| **config.py** | Parámetros globales: camas, médicos por día, penalizaciones, hiperparámetros AG |
| **pacientes.py** | Generación de lista de espera simulada con datos epidemiológicos de Lambayeque |
| **genetico.py** | Núcleo del AG: cromosoma, fitness, cruce, mutación, torneo, elitismo |
| **resultados.py** | Decodificación del plan óptimo, reporte en consola y 3 gráficas |
| **main.py** | Punto de entrada con argparse y barra de progreso animada |


Proceso de instalación:
pip install numpy pandas matplotlib tabulate

Scripts básicos:
# Ejecución básica (40 pacientes, 150 generaciones)
python main.py

# Con semilla fija (resultado reproducible)
python main.py --semilla 42

# Más pacientes y generaciones para resultados más refinados
python main.py --pacientes 55 --generaciones 200

# Sin gráficas (más rápido, solo consola)
python main.py --sin-graficas