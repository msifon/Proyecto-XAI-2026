"""
Análisis de importancia de boyas con TSMULE y CONFETTI, para un modelo.
"""

import sys,os, warnings, logging, pickle
#import contextlib
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

sys.path.insert(0, 'METHODS/ts-mule')

# Filter out all RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore") #supress warnings
logging.getLogger("stumpy").setLevel(logging.ERROR)

import keras
#from keras.models import load_model
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = ''

from importancia_boyas import plot_boya_importance_comparison
										
def save_pickle(variable, fname):
    with open(fname, 'wb') as f:
        pickle.dump(variable, f)

print('***** Paquetes cargados con éxito *****')
####### COMIENZO DEL PROCESO ##########

print('***** Iniciando análisis          *****')	
# Cargar resultados
with open(f'RESULTADOS_MULTI/TSMULE_resultados_all.pkl', 'rb') as file:
    TSMULE = pickle.load(file)
with open(f'RESULTADOS_MULTI/CONFETTI_resultados_all.pkl', 'rb') as file:
    CONFETTI = pickle.load(file)
print('***** Resutlados cargados         *****')

print('Comenzando a generar gráficos')
plot_boya_importance_comparison(TSMULE=TSMULE, CF=CONFETTI,
    feature_names=['Boya 1', 'Boya 2', 'Boya 3', 'Boya 4', 'Boya 5', 'Boya 6'],
    fig_fmt='png', save_path='RESULTADOS_MULTI/')



