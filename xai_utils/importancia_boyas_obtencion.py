"""
Explicaciones de importancia de boyas con TSMULE y CONFETTI, para un modelo.
"""

import sys,os, warnings, contextlib, logging, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.insert(0, 'METHODS/ts-mule')

# Filter out all RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore") #supress warnings
logging.getLogger("stumpy").setLevel(logging.ERROR)

import keras
from keras.models import load_model

import tsmule
print(f"ts-MULE cargado desde: {tsmule.__file__}")

os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = ''

from xai_utils.tsmule_analysis import analyze_with_tsmule
from xai_utils.confetti_analysis import analyze_with_confetti
from xai_utils.importancia_boyas import (plot_boya_importance_comparison, 
										plot_boya_modification_frequency)

def save_pickle(variable, fname):
    with open(fname, 'wb') as f:
        pickle.dump(variable, f)

print('***** Paquetes cargados con éxito *****')
####### COMIENZO DEL PROCESO ##########

start = datetime.now()

print('***** Iniciando proceso           *****')	
# Cargar Datos
with open('DATA/xtest_new.pickle', 'rb') as f:
	X_test = pickle.load(f)
with open('DATA/xtrain_new.pickle', 'rb') as f:
	X_train = pickle.load(f)
print('***** Set de datos cargados       *****')

# Cargar modelo
model = load_model('MODELS/transfer_learned_tsunami_classifier.keras')
print('***** Modelo cargado              *****')

N0, N1=[141, 500] #Instancia de inicio y fin

# TSMULE
n_runs=300 #300 mejor valor
n_samples=100 # 100 mejor valor
optim_ts = ''
if (n_runs==300) and (n_samples==100): optim_ts = '| Configuracion optima'

#CONFETTI
psize=100 #100
max_gen=200 #200
optim_cf = ''
if (psize==100) and (max_gen==200): optim_cf = '| Configuracion optima'

#TSMULE
TSMULE = []
# CONFETTI
CONFETTI = [] #Casos con contrafactual valido encontrado
indices_sin_cf = [] #casos en que no encuentra contrafactual (si es que pasa)

print('***********************************************************************')
print(f'  Iniciando explicaciones con TS-MULE y CONFETTI para {N1-N0} instancias')
print('***********************************************************************')
print(f' -TS-MULE : n_runs={n_runs} | n_samples={n_samples} {optim_ts}')
print(f' -CONFETTI:  psize={psize} | max_gen  ={max_gen} {optim_cf}')
print(f' -Instancia inicial: {N0}   | Instacia final: {N1}')
print(f' -Inicio: {str(datetime.now())} | Término estimado: {str(datetime.now()+timedelta(minutes=(N1-N0)*6.8))}')
print('-----------------------------------------------------------------------')


for n in range(N0, N1):
    print(f'*** Instancia: {n:04d} | {datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} ***')
    
    # TSMULE
    with open(os.devnull, 'w') as devnull: #Elimina stdout
        with contextlib.redirect_stdout(devnull):
                resultados_tsmule = analyze_with_tsmule(
                model=model,
                x=X_test[n],
                n_runs=n_runs,
                n_samples=n_samples,
                segmentation_method='bins-min', #'slopes-sorted'
                feature_names=['Boya 1', 'Boya 2', 'Boya 3', 'Boya 4', 'Boya 5', 'Boya 6'],
                save_path='RESULTADOS_TSMULE/'
                )
                TSMULE.append(resultados_tsmule)

    try:
    # CONFETTI
	    with open(os.devnull, 'w') as devnull:
	        with contextlib.redirect_stdout(devnull):
	            resultados_cf = analyze_with_confetti(
	                model_path_wrapped='MODELS/transfer_learned_tsunami_classifier_wrapped.keras',
	                model_path_original='MODELS/transfer_learned_tsunami_classifier.keras',
	                instance=X_test[n:n+1],
	                training_weights_path=None,
	                X_train=X_train,
	                n_partitions=5,
	                alpha=0.5,
	                theta=0.51,
	                optimize_sparsity=True,
	                population_size=psize,
	                maximum_number_of_generations=max_gen,
	                feature_names=['Boya 1', 'Boya 2', 'Boya 3', 'Boya 4', 'Boya 5', 'Boya 6'],
	                save_path=None,
	                use_cam_weights=False
	            )
	    
	    CONFETTI.append(resultados_cf)
	    
    except:
    	indices_sin_cf.append(n)
    	print(f'¡¡ Instancia {n} con problemas!! no se guarda')


	#Guardamos paso intermedio
    if (n%10==0) and (n>1):
    	print(f' ----- guardando resultados parciales ({n}) -----')
    	save_pickle(TSMULE, f'RESULTADOS_MULTI/TSMULE_resultados_{n}.pkl')
    	save_pickle(CONFETTI, f'RESULTADOS_MULTI/CONFETTI_resultados_{n}.pkl')

t_total = (datetime.now()-start)
total_min = t_total.total_seconds() / 60
print('-----------------------------------------------------------------------')
print(f'  Tiempo promedio por instancia: {total_min/(N1-N0):.1f} minutos')
print(f'  Proceso finalizado para instancias {N0} a {N1}:')
print(f'  TS-MULE :  {len(TSMULE)}')
print(f'  Con CF  :  {len(CONFETTI)}')
print(f'  Sin CF  :  {( (N1-N0)-len(CONFETTI) )}')
if indices_sin_cf:
    print(f'  Índices sin CF: {indices_sin_cf}')

print('-----------------------------------------------------------------------')
print('Guardando resultados finales :)')
# Guardar ts-MULE
save_pickle(TSMULE, f'RESULTADOS_MULTI/TSMULE_resultados_{N0}_{N1}.pkl')
save_pickle(CONFETTI, f'RESULTADOS_MULTI/CONFETTI_resultados_{N0}_{N1}.pkl')
save_pickle(indices_sin_cf, f'RESULTADOS_MULTI/CONFETTI_SIN_CF_{N0}_{N1}.pkl')
print(f'Resultados guardados.')
print('-----------------------------------------------------------------------')

print('Comenzando a generar gráficos')
plot_boya_importance_comparison(TSMULE=TSMULE, CF=CONFETTI,
    feature_names=['Boya 1', 'Boya 2', 'Boya 3', 'Boya 4', 'Boya 5', 'Boya 6'],
    fig_fmt='png', save_path='RESULTADOS_MULTI/')
plot_boya_modification_frequency(CONFETTI, feature_names=None, percentil=25, 
	fig_fmt='png', save_path='RESULTADOS_MULTI/')

end = datetime.now()
print(f'Tiempo total: {str(end-start)}')
print('*****************************')
print('****    VUELVA PRONTO   *****')
print('*****************************')
