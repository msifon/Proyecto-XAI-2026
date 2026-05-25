"""
tsmule_analysis.py
------------------
Función principal de análisis de relevancia de series de tiempo usando ts-MULE (LimeTS).

Bugs corregidos en ts-MULE respecto al repositorio original:
    1. segmentation_method='slope-max' renombrado a 'slopes-sorted' (nombre incorrecto)
    2. z_hat sin flatten en _explain() — shape (n,1) en vez de (n,) causaba coeficientes nulos
    3. Lasso(alpha=0.01) demasiado agresivo — reducido a alpha=0.0001
    4. mask_percentile() usaba '>' estricto — cambiado a '>=' con method='lower'

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
from datetime import datetime
import warnings
from sklearn.linear_model import Lasso


def analyze_with_tsmule(model, x, n_runs=100, segmentation_method='slopes-sorted',
                        partitions=5, win_length=4, alpha=0.0001, n_samples=100,
                        percentile=90, replace_method='zeros', random_seed=None,
                        feature_names=None, save_path=None):
    """
    Análisis de relevancia de series de tiempo usando ts-MULE (LimeTS).

    Args:
        model: Modelo de predicción (ej. keras model).
        x (ndarray): Serie de tiempo a analizar con shape (n_steps, n_features)
                     o (1, n_steps, n_features).
        n_runs (int): Número de iteraciones para promediar relevancia. Default: 100.
        segmentation_method (str): Método de segmentación. Default: 'slopes-sorted'.
            Opciones: 'slopes-sorted' | 'slopes-not-sorted' | 'bins-max' | 'bins-min'
        partitions (int): Número de particiones para la segmentación. Default: 5.
        win_length (int): Longitud de ventana para la segmentación. Default: 4.
        alpha (float): Regularización del kernel Lasso. Default: 0.0001.
        n_samples (int): Número de perturbaciones por iteración. Default: 100.
        percentile (float): Percentil para la máscara de relevancia. Default: 90.
        replace_method (str): Método de reemplazo en perturbación. Default: 'zeros'.
        random_seed (int): Semilla para reproducibilidad. Default: None.
        feature_names (list): Nombres de las features/boyas. Default: None.
        save_path (str): Ruta base para guardar resultados (.npy). Default: None.
            Ejemplo: 'RESULTADOS_TSMULE\\evento_1' generará:
                - evento_1_relevance_promedio.npy
                - evento_1_relevances_all.npy

    Returns:
        dict con keys:
            - 'relevance_promedio' (ndarray): shape (n_steps, n_features)
            - 'relevances_all'    (ndarray): shape (n_runs, n_steps, n_features)
            - 'x_perturbed'       (ndarray): shape (n_steps, n_features)
            - 'x_original'        (ndarray): shape (n_steps, n_features)
            - 'mask'              (ndarray): shape (n_steps, n_features)
    """
    from tsmule.xai.lime import LimeTS
    from tsmule.xai.evaluation import PerturbationAnalysis

    # Semilla de reproducibilidad
    if random_seed is not None:
        np.random.seed(random_seed)

    # Asegurar shape correcto (n_steps, n_features)
    if x.ndim == 3:
        x = x[0]

    n_steps, n_features = x.shape

    # Nombres de features por defecto
    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    # Función de predicción — compatible con Keras, maneja batch dimension
    def predict_fn(x_input):
        if x_input.ndim == 2:
            x_input = x_input[np.newaxis, ...]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pred = model(x_input, training=False)
        return float(pred.numpy().flatten()[0])

    # Configurar explainer
    explainer = LimeTS(
        n_samples=n_samples,
        win_length=win_length,
        partitions=partitions,
        kernel=Lasso(alpha=alpha)
    )

    pa = PerturbationAnalysis()

    # Calcular relevancia promediando n_runs iteraciones
    print(f"Calculando relevancia con {n_runs} iteraciones...")
    print(f"  Inicio: {datetime.now().strftime('%H:%M:%S')}")
    relevances_all = []

    for i in range(n_runs):
        if (i + 1) % 10 == 0:
            print(f"  Iteración {i+1}/{n_runs} — {datetime.now().strftime('%H:%M:%S')}")
        r = explainer.explain(x, predict_fn, segmentation_method=segmentation_method)
        relevances_all.append(r)

    relevance_promedio = np.mean(relevances_all, axis=0)
    print(f"¡Cálculo completado! Fin: {datetime.now().strftime('%H:%M:%S')}")

    # Calcular serie perturbada y máscara
    x_perturbed = next(pa.perturb(
        [x], [relevance_promedio],
        replace_method=replace_method,
        percentile=percentile
    ))
    mask = pa.mask_percentile(relevance_promedio, percentile)

    # Guardar resultados en disco
    if save_path:
        try:
            np.savez(f'{save_path}_resultados_tsmule.npz',
                     relevance_promedio=relevance_promedio,
                     relevances_all=np.array(relevances_all),
                     x_perturbed=x_perturbed,
                     x_original=x,
                     mask=mask)
            print(f"Resultados guardados en: {save_path}_resultados_tsmule.npz")
        except Exception as e:
            print(f"No se pudo guardar en disco: {e}")

    return {
        'relevance_promedio': relevance_promedio,
        'relevances_all':     np.array(relevances_all),
        'x_perturbed':        x_perturbed,
        'x_original':         x,
        'mask':               mask
    }
