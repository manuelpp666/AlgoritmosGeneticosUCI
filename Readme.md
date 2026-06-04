
#Opcional, el programa funciona igual# pip install matplotlib


# Ejecución completa (200 generaciones, 100 individuos)
python main.py

# Prueba rápida (verificar que funciona)
python main.py --gens 20 --pop 30

# Con 4 médicos de turno (máximo 24 camas activas)
python main.py --doctors 4

# Sin gráficos (solo consola + CSV)
python main.py --no-plots

# Todo junto
python main.py --gens 200 --pop 100 --doctors 5 --out-dir resultados