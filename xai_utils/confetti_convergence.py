"""
confetti_convergence.py
-----------------------
Análisis de convergencia para CONFETTI variando population_size y
maximum_number_of_generations.

Estrategia:
    - Grilla de combinaciones (population_size x generations)
    - Misma instancia para todas las combinaciones
    - Guarda resultados intermedios para retomar en caso de fallo
    - Métricas por combinación: encontró CF, diff_mean, diff_max,
      timesteps_modificados, n_counterfactuals, tiempo

Uso típico:
    from xai_utils import run_confetti_convergence, plot_confetti_convergence

    resultados_conv = run_confetti_convergence(
        model_path_wrapped='MODELS\\model_wrapped.keras',
        instance=instance,
        X_train=X_train,
        save_path='RESULTADOS_CONVERGENCIA\\'
    )
    plot_confetti_convergence(resultados_conv)

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import warnings
from datetime import datetime
from contextlib import contextmanager

warnings.filterwarnings('ignore', message='Unable to find acceptable character detection dependency')
warnings.filterwarnings('ignore', category=UserWarning,
                        message="The structure of `inputs` doesn't match")


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


def run_confetti_convergence(model_path_wrapped, instance, X_train,
                             population_sizes=None,
                             generations_list=None,
                             training_weights_path=None,
                             use_cam_weights=False,
                             n_partitions=5,
                             alpha=0.5,
                             theta=0.3,
                             optimize_sparsity=True,
                             save_path='RESULTADOS_CONVERGENCIA\\'):
    """
    Análisis de convergencia de CONFETTI variando population_size y
    maximum_number_of_generations.

    Args:
        model_path_wrapped (str):  Ruta al modelo wrapped para CONFETTI.
        instance (ndarray):        Instancia a explicar, shape (1, n_steps, n_features).
        X_train (ndarray):         Datos de entrenamiento.
        population_sizes (list):   Valores de population_size a probar.
                                   Default: [50, 100, 200, 300]
        generations_list (list):   Valores de generations a probar.
                                   Default: [50, 100, 200, 300]
        training_weights_path (str): Ruta a pesos CAM pre-calculados. Default: None.
        use_cam_weights (bool):    Si usar pesos CAM. Default: False.
        n_partitions (int):        Particiones temporales. Default: 5.
        alpha (float):             Balance confianza-sparsity. Default: 0.5.
        theta (float):             Umbral de cambio de clase. Default: 0.3.
        optimize_sparsity (bool):  Optimizar sparsity. Default: True.
        save_path (str):           Ruta base para guardar resultados intermedios.

    Returns:
        pd.DataFrame: Tabla de métricas por combinación de parámetros.
    """
    from confetti import CONFETTI

    if population_sizes is None:
        population_sizes = [50, 100, 200, 300]
    if generations_list is None:
        generations_list = [50, 100, 200, 300]

    # Crear carpeta si no existe
    os.makedirs(save_path, exist_ok=True)

    # Archivo de checkpoint
    checkpoint_path = os.path.join(save_path, 'convergence_checkpoint.pkl')

    # Cargar checkpoint si existe
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        resultados = checkpoint['resultados']
        combinaciones_completadas = checkpoint['completadas']
        print(f"Checkpoint encontrado — {len(combinaciones_completadas)} "
              f"combinaciones ya completadas.")
    else:
        resultados = []
        combinaciones_completadas = set()
        print("Iniciando análisis de convergencia desde cero.")

    # Cargar pesos CAM si corresponde
    if use_cam_weights and training_weights_path and os.path.exists(training_weights_path):
        training_weights_resized = np.load(training_weights_path)
        print(f"Pesos CAM cargados: {training_weights_resized.shape}")
    else:
        training_weights_resized = None

    # Cargar explainer
    with _suppress_stdout():
        explainer = CONFETTI(model_path=model_path_wrapped)

    total = len(population_sizes) * len(generations_list)
    completadas = len(combinaciones_completadas)

    print(f"\nTotal de combinaciones: {total}")
    print(f"Pendientes: {total - completadas}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}\n")
    print("-" * 60)

    for pop in population_sizes:
        for gen in generations_list:
            clave = (pop, gen)

            # Saltar si ya fue completada
            if clave in combinaciones_completadas:
                print(f"[SKIP] pop={pop:>4}, gen={gen:>4} — ya completada")
                continue

            t_inicio = datetime.now()

            try:
                with _suppress_stdout():
                    results = explainer.generate_counterfactuals(
                        instances_to_explain=instance,
                        reference_data=X_train,
                        reference_weights=training_weights_resized,
                        n_partitions=n_partitions,
                        alpha=alpha,
                        theta=theta,
                        optimize_sparsity=optimize_sparsity,
                        population_size=pop,
                        maximum_number_of_generations=gen,
                        verbose=False
                    )

                t_fin = datetime.now()
                elapsed = (t_fin - t_inicio).total_seconds() / 60  # en minutos

                # Extraer métricas
                found = results[0].best is not None
                if found:
                    all_cfs = results[0].all_counterfactuals
                    best_cf = results[0].best.counterfactual
                    diff    = np.abs(best_cf - instance[0])
                    mask    = diff.mean(axis=1) > 1e-6

                    # Métricas del mejor CF
                    best_diff_mean = diff.mean()
                    best_diff_max  = diff.max()
                    best_ts_mod    = int(np.sum(mask))

                    # Métricas promedio sobre todos los CFs
                    all_diffs = [np.abs(cf.counterfactual - instance[0]).mean()
                                 for cf in all_cfs]
                    mean_diff_all = np.mean(all_diffs)
                    n_cfs = len(all_cfs)
                else:
                    best_diff_mean = np.nan
                    best_diff_max  = np.nan
                    best_ts_mod    = np.nan
                    mean_diff_all  = np.nan
                    n_cfs          = 0

                fila = {
                    'population_size':       pop,
                    'generations':           gen,
                    'found_cf':              found,
                    'n_counterfactuals':     n_cfs,
                    'best_diff_mean':        best_diff_mean,
                    'best_diff_max':         best_diff_max,
                    'best_ts_modificados':   best_ts_mod,
                    'mean_diff_all_cfs':     mean_diff_all,
                    'tiempo_min':            round(elapsed, 2)
                }

            except Exception as e:
                t_fin = datetime.now()
                elapsed = (t_fin - t_inicio).total_seconds() / 60
                print(f"[ERROR] pop={pop:>4}, gen={gen:>4} — {e}")
                fila = {
                    'population_size':       pop,
                    'generations':           gen,
                    'found_cf':              False,
                    'n_counterfactuals':     0,
                    'best_diff_mean':        np.nan,
                    'best_diff_max':         np.nan,
                    'best_ts_modificados':   np.nan,
                    'mean_diff_all_cfs':     np.nan,
                    'tiempo_min':            round(elapsed, 2)
                }

            resultados.append(fila)
            combinaciones_completadas.add(clave)

            # Guardar checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'resultados':  resultados,
                    'completadas': combinaciones_completadas
                }, f)

            estado = '✓ CF encontrado' if fila['found_cf'] else '✗ Sin CF'
            print(f"[{estado}] pop={pop:>4}, gen={gen:>4} | "
                  f"n_cfs={fila['n_counterfactuals']:>3} | "
                  f"diff_mean={fila['best_diff_mean']:.5f} | "
                  f"ts_mod={fila['best_ts_modificados']} | "
                  f"t={fila['tiempo_min']:.1f} min | "
                  f"fin: {t_fin.strftime('%H:%M:%S')}")

    # Construir DataFrame final
    df = pd.DataFrame(resultados)
    df = df.sort_values(['population_size', 'generations']).reset_index(drop=True)

    # Guardar tabla final
    df.to_csv(os.path.join(save_path, 'convergence_results.csv'), index=False)
    with open(os.path.join(save_path, 'convergence_results.pkl'), 'wb') as f:
        pickle.dump(df, f)

    print(f"\n{'='*60}")
    print(f"Análisis completado: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Resultados guardados en: {save_path}")
    print(f"{'='*60}")

    return df
