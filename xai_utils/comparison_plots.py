"""
comparison_plots.py
-------------------
Funciones de graficación comparativa entre ts-MULE y CONFETTI.

Funciones disponibles:
    - plot_comparison_summary:    Resumen comparativo por boya (barras)
    - plot_comparison_regions:    Ventanas temporales sobre serie original
    - plot_comparison_heatmaps:   Heatmaps comparativos lado a lado

Parámetro común cf_source:
    - 'best'  : usa el mejor contrafactual (best_cf)
    - 'mean'  : usa el promedio de todos los contrafactuales

Uso típico:
    from xai_utils import (plot_comparison_summary, plot_comparison_regions,
                           plot_comparison_heatmaps)

    plot_comparison_summary(resultados_tsmule, resultados_cf)
    plot_comparison_regions(resultados_tsmule, resultados_cf, cf_source='best')
    plot_comparison_heatmaps(resultados_tsmule, resultados_cf, cf_source='mean')

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import matplotlib.pyplot as plt


def _get_cf_array(resultados_cf, cf_source):
    """
    Obtiene el array del contrafactual según cf_source.

    Args:
        resultados_cf (dict): Diccionario retornado por analyze_with_confetti().
        cf_source (str):      'best' o 'mean'.

    Returns:
        ndarray: shape (n_steps, n_features)
    """
    if cf_source == 'best':
        return resultados_cf['best_cf']
    elif cf_source == 'mean':
        return np.mean([s['cf'] for s in resultados_cf['cf_stats']], axis=0)
    else:
        raise ValueError(f"cf_source debe ser 'best' o 'mean', recibido: '{cf_source}'")


def plot_comparison_summary(resultados_tsmule, resultados_cf,
                            feature_names=None, fig_fmt='png', save_path=None):
    """
    Resumen comparativo de relevancia/modificación por boya entre ts-MULE y CONFETTI.
    Layout 2x2 con valores normalizados entre 0 y 1 para comparación directa.

    Fila 1: Media  — ts-MULE (izq) | CONFETTI (der)
    Fila 2: Máxima — ts-MULE (izq) | CONFETTI (der)

    Args:
        resultados_tsmule (dict): Diccionario retornado por analyze_with_tsmule().
        resultados_cf (dict):     Diccionario retornado por analyze_with_confetti().
        feature_names (list):     Nombres de las features/boyas. Default: None.
        fig_fmt (str):            Formato de figura: 'pdf' o 'png'. Default: 'pdf'.
        save_path (str):          Ruta base para guardar figura. Default: None.
    """
    relevance  = resultados_tsmule['relevance_promedio']
    instance   = resultados_cf['instance']
    cf_stats   = resultados_cf['cf_stats']
    n_features = relevance.shape[1]

    if feature_names is None:
        feature_names = resultados_cf.get('feature_names',
                                          [f'Boya {i+1}' for i in range(n_features)])

    # ts-MULE: relevancia absoluta por boya
    rel_mean = np.mean(np.abs(relevance), axis=0)
    rel_max  = np.max(np.abs(relevance),  axis=0)

    # CONFETTI: modificación por boya sobre todos los CFs
    diffs_por_boya = np.array([
        np.mean(np.abs(s['cf'] - instance), axis=0)
        for s in cf_stats if s['t_inicio'] is not None
    ])
    cf_mean = diffs_por_boya.mean(axis=0)
    cf_max  = diffs_por_boya.max(axis=0)

    # Normalizar entre 0 y 1 dentro de cada métrica
    def norm(x):
        return x / x.max() if x.max() > 0 else x

    rel_mean_norm = norm(rel_mean)
    rel_max_norm  = norm(rel_max)
    cf_mean_norm  = norm(cf_mean)
    cf_max_norm   = norm(cf_max)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)

    # Fila 1: Media
    axes[0, 0].barh(feature_names, rel_mean_norm, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('ts-MULE — Relevancia media')
    axes[0, 0].set_xlabel('Relevancia normalizada')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].set_xlim(0, 1.05)

    axes[0, 1].barh(feature_names, cf_mean_norm, color='orange', alpha=0.8)
    axes[0, 1].set_title('CONFETTI — Modificación media')
    axes[0, 1].set_xlabel('Modificación normalizada')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].set_xlim(0, 1.05)

    # Fila 2: Máxima
    axes[1, 0].barh(feature_names, rel_max_norm, color='steelblue', alpha=0.5)
    axes[1, 0].set_title('ts-MULE — Relevancia máxima')
    axes[1, 0].set_xlabel('Relevancia normalizada')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].set_xlim(0, 1.05)

    axes[1, 1].barh(feature_names, cf_max_norm, color='orange', alpha=0.5)
    axes[1, 1].set_title('CONFETTI — Modificación máxima')
    axes[1, 1].set_xlabel('Modificación normalizada')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].set_xlim(0, 1.05)

    fig.suptitle('Resumen comparativo por boya — ts-MULE vs CONFETTI\n'
                 '(valores normalizados entre 0 y 1)', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}comparacion_resumen.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_comparacion_resumen.{fig_fmt}")

    plt.show()

def plot_comparison_regions(resultados_tsmule, resultados_cf,
                            cf_source='best', percentile=90,
                            feature_names=None, fig_fmt='png', save_path=None):
    """
    Ventanas temporales relevantes de cada método superpuestas sobre la serie original.
    Layout de 2 columnas: boyas 1-3 (izquierda) y boyas 4-6 (derecha).

    Para cada boya muestra:
        - Serie original (negro)
        - Regiones relevantes ts-MULE (azul)
        - Regiones modificadas CONFETTI (naranja)

    Args:
        resultados_tsmule (dict): Diccionario retornado por analyze_with_tsmule().
        resultados_cf (dict):     Diccionario retornado por analyze_with_confetti().
        cf_source (str):          'best' o 'mean'. Default: 'best'.
        percentile (float):       Percentil para máscara ts-MULE. Default: 90.
        feature_names (list):     Nombres de las features/boyas. Default: None.
        fig_fmt (str):            Formato de figura: 'pdf' o 'png'. Default: 'pdf'.
        save_path (str):          Ruta base para guardar figura. Default: None.
    """
    x_original  = resultados_tsmule['x_original']
    x_perturbed = resultados_tsmule['x_perturbed']
    instance    = resultados_cf['instance']
    cf_array    = _get_cf_array(resultados_cf, cf_source)

    n_steps, n_features = x_original.shape
    n_rows = (n_features + 1) // 2  # filas necesarias para 2 columnas

    if feature_names is None:
        feature_names = resultados_cf.get('feature_names',
                                          [f'Boya {i+1}' for i in range(n_features)])

    # Máscaras
    mask_tsmule = np.abs(x_original - x_perturbed) > 1e-10
    mask_cf     = np.abs(cf_array - instance) > 1e-6

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 3), sharex=True)

    for i in range(n_features):
        row = i % n_rows
        col = i // n_rows
        ax  = axes[row, col]

        ax.plot(x_original[:, i], color='black', linewidth=1.5,
                label='Serie original', zorder=3)
        ax.fill_between(range(n_steps),
                        ax.get_ylim()[0], ax.get_ylim()[1],
                        where=mask_tsmule[:, i],
                        alpha=0.3, color='steelblue', label='Relevante ts-MULE')
        ax.fill_between(range(n_steps),
                        ax.get_ylim()[0], ax.get_ylim()[1],
                        where=mask_cf[:, i],
                        alpha=0.3, color='orange', label='Modificado CONFETTI')
        ax.set_title(feature_names[i])
        ax.set_ylabel('Amplitud')
        ax.legend(loc='upper right', fontsize=8)

    # Etiqueta eje x en la última fila de cada columna
    for col in range(2):
        axes[n_rows - 1, col].set_xlabel('Timesteps')

    cf_label = 'mejor CF' if cf_source == 'best' else 'promedio CFs'
    fig.suptitle(f'Regiones relevantes por método — ts-MULE vs CONFETTI ({cf_label})',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}comparacion_regiones.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_comparacion_regiones.{fig_fmt}")

    plt.show()


def plot_comparison_heatmaps(resultados_tsmule, resultados_cf,
                             cf_source='best', feature_names=None,
                             fig_fmt='png', save_path=None, cmap='Reds'):
    """
    Heatmaps comparativos lado a lado compartiendo eje y (boyas).
    Ambos mapas normalizados entre 0 y 1 y con el mismo colormap 'Reds'
    para facilitar la comparación visual directa.

    Izquierda: relevancia absoluta ts-MULE  (normalizada)
    Derecha:   modificación CONFETTI        (normalizada)

    Args:
        resultados_tsmule (dict): Diccionario retornado por analyze_with_tsmule().
        resultados_cf (dict):     Diccionario retornado por analyze_with_confetti().
        cf_source (str):          'best' o 'mean'. Default: 'best'.
        feature_names (list):     Nombres de las features/boyas. Default: None.
        fig_fmt (str):            Formato de figura: 'pdf' o 'png'. Default: 'pdf'.
        save_path (str):          Ruta base para guardar figura. Default: None.
    """
    relevance  = resultados_tsmule['relevance_promedio']
    instance   = resultados_cf['instance']
    cf_array   = _get_cf_array(resultados_cf, cf_source)
    n_features = relevance.shape[1]

    if feature_names is None:
        feature_names = resultados_cf.get('feature_names',
                                          [f'Boya {i+1}' for i in range(n_features)])

    # Valor absoluto y normalización entre 0 y 1
    def norm(x):
        x_abs = np.abs(x)
        return x_abs / x_abs.max() if x_abs.max() > 0 else x_abs

    relevance_norm = norm(relevance)
    cf_diff_norm   = norm(cf_array - instance)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

    # ts-MULE
    im0 = axes[0].imshow(relevance_norm.T, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title('Relevancia absoluta normalizada — ts-MULE')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Boyas')
    axes[0].set_yticks(range(n_features))
    axes[0].set_yticklabels(feature_names)
    fig.colorbar(im0, ax=axes[0], label='Relevancia normalizada')

    # CONFETTI
    cf_label = 'mejor CF' if cf_source == 'best' else 'promedio CFs'
    im1 = axes[1].imshow(cf_diff_norm.T, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f'Modificación normalizada — CONFETTI ({cf_label})')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_yticks(range(n_features))
    axes[1].set_yticklabels(feature_names)
    fig.colorbar(im1, ax=axes[1], label='Modificación normalizada')

    fig.suptitle('Heatmaps comparativos normalizados — ts-MULE vs CONFETTI',
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}comparacion_heatmaps.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_comparacion_heatmaps.{fig_fmt}")

    plt.show()
