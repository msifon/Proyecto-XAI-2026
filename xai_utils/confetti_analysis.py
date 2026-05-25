"""
confetti_analysis.py
--------------------
Función principal de análisis de contrafactuales para series de tiempo usando CONFETTI.

Nota sobre los modelos:
    CONFETTI requiere dos versiones del mismo modelo:
    - model_path_original: modelo original, usado para calcular los pesos CAM
    - model_path_wrapped:  modelo con capa de salida wrapped (BinaryOutputLayer),
                           requerido por el explainer CONFETTI

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import os, sys, pickle
from contextlib import contextmanager

import keras
from keras.models import load_model
from confetti import CONFETTI
from confetti.attribution import cam
import warnings
warnings.filterwarnings('ignore', message='Unable to find acceptable character detection dependency')
warnings.filterwarnings('ignore', category=UserWarning, 
                        message="The structure of `inputs` doesn't match")

@keras.saving.register_keras_serializable()
class BinaryOutputLayer(keras.layers.Layer):
    """Capa de salida wrapped requerida por CONFETTI."""
    def call(self, x):
        prob_clase_1 = keras.ops.squeeze(x, axis=-1)
        prob_clase_0 = 1.0 - prob_clase_1
        return keras.ops.stack([prob_clase_0, prob_clase_1], axis=1)

@contextmanager
def _suppress_stdout():
    """Suprime temporalmente la salida estándar."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def _resize_cam_weights(cam_weights, target_length):
    """Redimensiona pesos CAM al largo objetivo usando interpolación lineal."""
    x_orig = np.linspace(0, 1, len(cam_weights))
    x_target = np.linspace(0, 1, target_length)
    return interp1d(x_orig, cam_weights, kind='linear')(x_target)


def _compute_cf_stats(all_cfs, instance):
    """
    Calcula estadísticas para cada contrafactual generado.

    Returns:
        list of dict con keys: index, cf, label, diff_max, diff_mean,
                               timesteps_modificados, t_inicio, t_fin
    """
    cf_stats = []
    for i, cf in enumerate(all_cfs):
        cf_array = cf.counterfactual
        diff = np.abs(cf_array - instance[0])
        mask = diff.mean(axis=1) > 1e-6
        cf_stats.append({
            'index':                i,
            'cf':                   cf_array,
            'label':                cf.label,
            'diff_max':             diff.max(),
            'diff_mean':            diff.mean(),
            'timesteps_modificados': int(np.sum(mask)),
            't_inicio':             int(np.where(mask)[0][0])  if np.any(mask) else None,
            't_fin':                int(np.where(mask)[0][-1]) if np.any(mask) else None,
        })
    return cf_stats


def analyze_with_confetti(model_path_wrapped, model_path_original, instance, X_train,
                          n_partitions=5, alpha=0.5, theta=0.5,
                          optimize_sparsity=True, population_size=300,
                          maximum_number_of_generations=300,
                          feature_names=None, save_path=None, training_weights_path=None, 
                          verbose=False, use_cam_weights=False):
    """
    Análisis de contrafactuales de series de tiempo usando CONFETTI.

    Args:
        model_path_wrapped (str):  Ruta al modelo wrapped (para CONFETTI).
        model_path_original (str): Ruta al modelo original (para CAM).
        instance (ndarray):        Serie a explicar, shape (1, n_steps, n_features).
        X_train (ndarray):         Datos de entrenamiento, shape (n, n_steps, n_features).
        n_partitions (int):        Segmentos temporales para CONFETTI. Default: 5.
        alpha (float):             Balance confianza-sparsity. Default: 0.5.
        theta (float):             Umbral de cambio de clase. Default: 0.5.
        optimize_sparsity (bool):  Optimizar sparsity. Default: True.
        population_size (int):     Tamaño de población del algoritmo genético. Default: 300.
        maximum_number_of_generations (int): Máximo de generaciones. Default: 300.
        feature_names (list):      Nombres de las features/boyas. Default: None.
        save_path (str):           Ruta base para guardar resultados (.npy). Default: None.

    Returns:
        dict con keys:
            - 'cf_stats'              (list):    estadísticas de cada contrafactual
            - 'cf_stats_sorted_diff'  (list):    ordenados por diff_mean (menor a mayor)
            - 'cf_stats_sorted_ts'    (list):    ordenados por timesteps_modificados
            - 'cf_stats_sorted_region'(list):    ordenados por t_inicio
            - 'best_cf'               (ndarray): mejor contrafactual, shape (n_steps, n_features)
            - 'instance'              (ndarray): instancia original, shape (n_steps, n_features)
            - 'feature_names'         (list):    nombres de features usados
            - 'results'               (obj):     objeto raw de CONFETTI
    """
    n_steps = instance.shape[1]
    n_features = instance.shape[2]

    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    # Cargar modelo original y calcular pesos CAM
    print(f"Cargando modelo original y calculando pesos CAM...")
    print(f"  Inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    # Cargar o calcular pesos CAM
    if not use_cam_weights:
        print("Usando reference_weights=None")
        training_weights_resized = None
    elif training_weights_path and os.path.exists(training_weights_path):
        print(f"Cargando pesos CAM desde: {training_weights_path}")
        training_weights_resized = np.load(training_weights_path)
        print(f"  Shape cargado: {training_weights_resized.shape}")
    else:
        print(f"Calculando pesos CAM...")
        model_original = load_model(model_path_original)
        training_weights = cam(model_original, X_train)
        training_weights_resized = np.array([
            _resize_cam_weights(w, n_steps) for w in training_weights
        ])
        print(f"  Shape: {training_weights_resized.shape}")
        if training_weights_path:
            np.save(training_weights_path, training_weights_resized)
            print(f"  Pesos CAM guardados en: {training_weights_path}")
        # print(f"  Pesos CAM shape original:       {training_weights.shape}")
        print(f"  Pesos CAM shape redimensionado: {training_weights_resized.shape}")

    # Cargar explainer CONFETTI con modelo wrapped
    print(f"\nCargando explainer CONFETTI...")
    explainer = CONFETTI(model_path=model_path_wrapped)

    # Generar contrafactuales
    print(f"\nGenerando contrafactuales...")
    print(f"  Parámetros: n_partitions={n_partitions}, alpha={alpha}, theta={theta}, "
          f"optimize_sparsity={optimize_sparsity}, population_size={population_size}")
    with _suppress_stdout():
        results = explainer.generate_counterfactuals(
            instances_to_explain=instance,
            reference_data=X_train,
            reference_weights=training_weights_resized if training_weights_resized is not None else None,
            n_partitions=n_partitions,
            alpha=alpha,
            theta=theta,
            optimize_sparsity=optimize_sparsity,
            population_size=population_size,
            maximum_number_of_generations=maximum_number_of_generations,
            verbose=verbose
        )
    print(f"¡Generación completada! Fin: {datetime.now().strftime('%H:%M:%S')}")

    # Verificar resultado
    if results[0].best is None:
        print("ADVERTENCIA: No se encontró contrafactual válido. "
              "Considera reducir theta o aumentar population_size.")
        return None

    best_cf = results[0].best.counterfactual
    all_cfs = results[0].all_counterfactuals
    print(f"Contrafactuales generados: {len(all_cfs)}")

    # Calcular estadísticas
    cf_stats = _compute_cf_stats(all_cfs, instance)

    # Ordenar por diferentes criterios
    cf_stats_sorted_diff = sorted(cf_stats, key=lambda x: x['diff_mean'])
    cf_stats_sorted_ts = sorted(
        [s for s in cf_stats if s['t_inicio'] is not None],
        key=lambda x: x['timesteps_modificados']
    )
    cf_stats_sorted_region = sorted(
        [s for s in cf_stats if s['t_inicio'] is not None],
        key=lambda x: x['t_inicio']
    )

    # Guardar resultados en disco
    if save_path:
        try:
            with open(f'{save_path}_resultados_cf.pkl', 'wb') as f:
                pickle.dump({
                    'cf_stats':               cf_stats,
                    'cf_stats_sorted_diff':   cf_stats_sorted_diff,
                    'cf_stats_sorted_ts':     cf_stats_sorted_ts,
                    'cf_stats_sorted_region': cf_stats_sorted_region,
                    'best_cf':                best_cf,
                    'instance':               instance[0],
                    'feature_names':          feature_names
                }, f)
            print(f"Resultados guardados en: {save_path}_resultados_cf.pkl")
        except Exception as e:
            print(f"No se pudo guardar en disco: {e}")

    return {
        'cf_stats':               cf_stats,
        'cf_stats_sorted_diff':   cf_stats_sorted_diff,
        'cf_stats_sorted_ts':     cf_stats_sorted_ts,
        'cf_stats_sorted_region': cf_stats_sorted_region,
        'best_cf':                best_cf,
        'instance':               instance[0],
        'feature_names':          feature_names,
        'results':                results
    }
