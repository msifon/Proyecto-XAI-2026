"""
tsmule_convergence.py
---------------------
Análisis de convergencia para ts-MULE variando n_runs.

Estrategia:
    - Acumula iteraciones progresivamente (sin recalcular desde cero)
    - Guarda checkpoints intermedios para retomar en caso de fallo
    - Métricas por bloque: diff_media respecto al bloque anterior,
      rango de relevancia, % de valores no nulos, tiempo

Uso típico:
    from xai_utils import run_tsmule_convergence, plot_tsmule_convergence

    df_conv = run_tsmule_convergence(
        model=model,
        x=X_nuevo,
        runs_to_test=[10, 25, 50, 100, 200, 300, 400, 500],
        save_path='RESULTADOS_CONVERGENCIA\\'
    )

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
from datetime import datetime
from sklearn.linear_model import Lasso


def run_tsmule_convergence(model, x,
                           runs_to_test=None,
                           segmentation_method='slopes-sorted',
                           partitions=5,
                           win_length=4,
                           alpha=0.0001,
                           n_samples=100,
                           save_path='RESULTADOS_CONVERGENCIA\\'):
    """
    Análisis de convergencia de ts-MULE variando n_runs acumulativamente.

    Args:
        model:                  Modelo Keras de predicción.
        x (ndarray):            Serie a analizar, shape (n_steps, n_features)
                                o (1, n_steps, n_features).
        runs_to_test (list):    Valores acumulativos de n_runs a evaluar.
                                Default: [10, 25, 50, 100, 200, 300, 400, 500]
        segmentation_method (str): Método de segmentación. Default: 'slopes-sorted'.
        partitions (int):       Número de particiones. Default: 5.
        win_length (int):       Longitud de ventana. Default: 4.
        alpha (float):          Regularización Lasso. Default: 0.0001.
        n_samples (int):        Perturbaciones por iteración. Default: 100.
        save_path (str):        Ruta base para guardar resultados intermedios.

    Returns:
        pd.DataFrame: Tabla de métricas por bloque de n_runs.
    """
    from tsmule.xai.lime import LimeTS

    if runs_to_test is None:
        runs_to_test = [10, 25, 50, 100, 200, 300, 400, 500]

    # Crear carpeta si no existe
    os.makedirs(save_path, exist_ok=True)

    # Archivo de checkpoint
    checkpoint_path = os.path.join(save_path, 'tsmule_convergence_checkpoint.pkl')

    # Asegurar shape correcto (n_steps, n_features)
    if x.ndim == 3:
        x = x[0]

    # Función de predicción
    input_name = model.input_names[0] if hasattr(model, 'input_names') else None

    def predict_fn(x_input):
        if x_input.ndim == 2:
            x_input = x_input[np.newaxis, ...]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if input_name:
                pred = model({input_name: x_input}, training=False)
            else:
                pred = model(x_input, training=False)
        return float(pred.numpy().flatten()[0])

    # Configurar explainer
    explainer = LimeTS(
        n_samples=n_samples,
        win_length=win_length,
        partitions=partitions,
        kernel=Lasso(alpha=alpha)
    )

    # Cargar checkpoint si existe
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        relevances_acum = checkpoint['relevances_acum']
        resultados      = checkpoint['resultados']
        prev_n          = checkpoint['prev_n']
        convergence     = checkpoint['convergence']
        print(f"Checkpoint encontrado — {prev_n} iteraciones ya completadas.")
    else:
        relevances_acum = []
        resultados      = []
        prev_n          = 0
        convergence     = []
        print("Iniciando análisis de convergencia desde cero.")

    # Filtrar bloques ya completados
    runs_pendientes = [n for n in runs_to_test if n > prev_n]

    if not runs_pendientes:
        print("Todos los bloques ya están completados.")
        return pd.DataFrame(resultados)

    total_pendiente = runs_to_test[-1] - prev_n
    print(f"\nIteraciones pendientes: {total_pendiente}")
    print(f"Bloques pendientes: {runs_pendientes}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}\n")
    print("-" * 60)

    for n in runs_pendientes:
        t_inicio = datetime.now()

        # Solo ejecutar iteraciones adicionales
        for i in range(n - prev_n):
            r = explainer.explain(x, predict_fn,
                                  segmentation_method=segmentation_method)
            relevances_acum.append(r)

        t_fin = datetime.now()
        elapsed = (t_fin - t_inicio).total_seconds() / 60

        # Calcular promedio acumulado
        prom = np.mean(relevances_acum, axis=0)
        convergence.append(prom.copy())

        # Métricas
        diff = np.mean(np.abs(convergence[-1] - convergence[-2])) \
               if len(convergence) > 1 else np.nan
        rango_min   = prom.min()
        rango_max   = prom.max()
        pct_nonzero = 100 * np.count_nonzero(prom) / prom.size

        fila = {
            'n_runs':       n,
            'diff_media':   round(diff, 6)   if not np.isnan(diff) else np.nan,
            'rango_min':    round(rango_min, 6),
            'rango_max':    round(rango_max, 6),
            'pct_no_nulos': round(pct_nonzero, 2),
            'tiempo_min':   round(elapsed, 2)
        }
        resultados.append(fila)

        # Guardar checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'relevances_acum': relevances_acum,
                'resultados':      resultados,
                'prev_n':          n,
                'convergence':     convergence
            }, f)

        # Guardar relevancia promedio del bloque
        np.save(os.path.join(save_path, f'tsmule_relevance_n{n}.npy'), prom)

        print(f"n={n:>4} | diff={fila['diff_media']} | "
              f"rango=[{fila['rango_min']:.4f}, {fila['rango_max']:.4f}] | "
              f"no_nulos={fila['pct_no_nulos']:.1f}% | "
              f"t={fila['tiempo_min']:.1f} min | "
              f"fin: {t_fin.strftime('%H:%M:%S')}")

        prev_n = n

    # Guardar relevances acumulados y tabla final
    np.save(os.path.join(save_path, 'tsmule_relevances_acumulados.npy'),
            np.array(relevances_acum))

    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(save_path, 'tsmule_convergence_results.csv'), index=False)
    with open(os.path.join(save_path, 'tsmule_convergence_results.pkl'), 'wb') as f:
        pickle.dump(df, f)

    print(f"\n{'='*60}")
    print(f"Análisis completado: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Resultados guardados en: {save_path}")
    print(f"{'='*60}")

    return df
