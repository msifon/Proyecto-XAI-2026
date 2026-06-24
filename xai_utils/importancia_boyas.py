"""
multi_analysis.py
-----------------
Funciones para análisis de importancia de boyas sobre múltiples instancias,
combinando resultados de ts-MULE y CONFETTI.

Funciones disponibles:
    - compute_boya_importance_tsmule:       Importancia por boya desde ts-MULE
    - compute_boya_importance:              Importancia por boya desde CONFETTI
    - compute_boya_modification_frequency:  Frecuencia de modificación desde CONFETTI
    - plot_boya_importance_tsmule:          Gráfico importancia ts-MULE
    - plot_boya_importance_confetti:        Gráfico importancia CONFETTI
    - plot_boya_importance_comparison:      Gráfico comparativo 2x2
    - plot_boya_modification_frequency:     Gráfico frecuencia de modificación CONFETTI

Uso típico:
    from xai_utils.multi_analysis import (
        compute_boya_importance_tsmule,
        compute_boya_importance,
        compute_boya_modification_frequency,
        plot_boya_importance_tsmule,
        plot_boya_importance_confetti,
        plot_boya_importance_comparison,
        plot_boya_modification_frequency
    )

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ==============================================================
# Funciones de cómputo
# ==============================================================

def compute_boya_importance_tsmule(TSMULE, feature_names=None):
    """
    Calcula importancia por boya sobre todas las instancias analizadas con ts-MULE.

    Args:
        TSMULE (list):        Lista de diccionarios retornados por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.

    Returns:
        tuple: (imp_mean, imp_max, feature_names)
            - imp_mean: relevancia absoluta media por boya
            - imp_max:  relevancia absoluta máxima por boya
    """
    n_features = TSMULE[0]['relevance_promedio'].shape[1]
    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    imp_mean_all = []
    imp_max_all  = []

    for resultado in TSMULE:
        if resultado is None:
            continue
        relevance = resultado['relevance_promedio']
        imp_mean_all.append(np.mean(np.abs(relevance), axis=0))
        imp_max_all.append(np.max(np.abs(relevance),   axis=0))

    imp_mean = np.mean(imp_mean_all, axis=0)
    imp_max  = np.mean(imp_max_all,  axis=0)

    return imp_mean, imp_max, feature_names


def compute_boya_importance(CF, feature_names=None):
    """
    Calcula importancia por boya sobre todas las instancias analizadas con CONFETTI.

    Args:
        CF (list):            Lista de diccionarios retornados por analyze_with_confetti().
        feature_names (list): Nombres de las features/boyas. Default: None.

    Returns:
        tuple: (imp_best, imp_mean, feature_names)
            - imp_best: modificación media del mejor CF por boya
            - imp_mean: modificación media promedio de todos los CFs por boya
    """
    n_features = CF[0]['instance'].shape[1]
    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    diff_best_all = []
    diff_mean_all = []

    for resultado in CF:
        if resultado is None:
            continue
        instance = resultado['instance']
        best_cf  = resultado['best_cf']
        cf_stats = resultado['cf_stats']

        diff_best_all.append(
            np.mean(np.abs(best_cf - instance), axis=0)
        )
        diff_mean_all.append(np.mean([
            np.mean(np.abs(s['cf'] - instance), axis=0)
            for s in cf_stats
        ], axis=0))

    imp_best = np.mean(diff_best_all, axis=0)
    imp_mean = np.mean(diff_mean_all, axis=0)

    return imp_best, imp_mean, feature_names


def compute_boya_modification_frequency(CF, feature_names=None, threshold=1e-6):
    """
    Calcula la frecuencia con que cada boya es modificada en los contrafactuales.

    Args:
        CF (list):            Lista de diccionarios retornados por analyze_with_confetti().
        feature_names (list): Nombres de las features/boyas. Default: None.
        threshold (float):    Umbral para considerar una modificación. Default: 1e-6.

    Returns:
        tuple: (freq_best, freq_mean, feature_names)
            - freq_best: frecuencia de modificación en el mejor CF
            - freq_mean: frecuencia promedio de modificación en todos los CFs
    """
    n_features = CF[0]['instance'].shape[1]
    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    freq_best_all = []
    freq_mean_all = []

    for resultado in CF:
        if resultado is None:
            continue
        instance = resultado['instance']
        best_cf  = resultado['best_cf']
        cf_stats = resultado['cf_stats']

        diff_best = np.mean(np.abs(best_cf - instance), axis=0)
        freq_best_all.append((diff_best > threshold).astype(float))

        modifica = np.array([
            (np.mean(np.abs(s['cf'] - instance), axis=0) > threshold).astype(float)
            for s in cf_stats
        ])
        freq_mean_all.append(modifica.mean(axis=0))

    freq_best = np.mean(freq_best_all, axis=0)
    freq_mean = np.mean(freq_mean_all, axis=0)

    return freq_best, freq_mean, feature_names


def compute_modification_threshold(CF, percentil=25):
    """
    Calcula el umbral de modificación basado en un percentil de las diferencias
    observadas en los mejores contrafactuales.

    Args:
        CF (list):      Lista de diccionarios retornados por analyze_with_confetti().
        percentil (int): Percentil a usar como umbral. Default: 25.

    Returns:
        float: Umbral calculado.
    """
    diffs = []
    for resultado in CF:
        if resultado is None:
            continue
        diff = np.mean(np.abs(resultado['best_cf'] - resultado['instance']), axis=0)
        diffs.extend(diff.tolist())

    threshold = np.percentile(diffs, percentil)
    print(f"Percentil {percentil}: {threshold:.6f}")
    return threshold


# ==============================================================
# Funciones de graficación
# ==============================================================

def _add_bar_labels(ax, bars, color='white'):
    """Agrega etiquetas de porcentaje dentro de las barras."""
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 3, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', va='center', ha='left',
                fontsize=9, color=color, fontweight='bold')


def plot_boya_importance_tsmule(TSMULE, feature_names=None,
                                 fig_fmt='png', save_path=None):
    """
    Gráfico de importancia por boya sobre todas las instancias analizadas con ts-MULE.
    Layout 1x2: relevancia media (izq) y máxima (der).
    Valores normalizados para que sumen 100%.

    Args:
        TSMULE (list):        Lista de diccionarios retornados por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.
        fig_fmt (str):        Formato de figura. Default: 'png'.
        save_path (str):      Ruta base para guardar figura. Default: None.
    """
    imp_mean, imp_max, feature_names = compute_boya_importance_tsmule(
        TSMULE, feature_names=feature_names
    )

    imp_mean_norm = imp_mean / imp_mean.sum() * 100
    imp_max_norm  = imp_max  / imp_max.sum()  * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

    bars0 = axes[0].barh(feature_names, imp_mean_norm, color='#4CAF50', alpha=0.85)
    _add_bar_labels(axes[0], bars0)
    axes[0].set_xlabel('Importancia relativa (%)')
    axes[0].set_title('Importancia por boya — Relevancia media')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].xaxis.set_major_formatter(PercentFormatter())

    bars1 = axes[1].barh(feature_names, imp_max_norm, color='#1B5E20', alpha=0.85)
    _add_bar_labels(axes[1], bars1)
    axes[1].set_xlabel('Importancia relativa (%)')
    axes[1].set_title('Importancia por boya — Relevancia máxima')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].xaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(f'Importancia de boyas — ts-MULE ({len(TSMULE)} instancias)',
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}_importancia_boyas_tsmule.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_importancia_boyas_tsmule.{fig_fmt}")

    plt.show()


def plot_boya_importance_confetti(CF, feature_names=None,
                                   fig_fmt='png', save_path=None):
    """
    Gráfico de importancia por boya sobre todas las instancias analizadas con CONFETTI.
    Layout 1x2: mejor CF (izq) y promedio CFs (der).
    Valores normalizados para que sumen 100%.

    Args:
        CF (list):            Lista de diccionarios retornados por analyze_with_confetti().
        feature_names (list): Nombres de las features/boyas. Default: None.
        fig_fmt (str):        Formato de figura. Default: 'png'.
        save_path (str):      Ruta base para guardar figura. Default: None.
    """
    imp_best, imp_mean, feature_names = compute_boya_importance(
        CF, feature_names=feature_names
    )

    imp_best_norm = imp_best / imp_best.sum() * 100
    imp_mean_norm = imp_mean / imp_mean.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

    bars0 = axes[0].barh(feature_names, imp_best_norm, color='#9C27B0', alpha=0.85)
    _add_bar_labels(axes[0], bars0)
    axes[0].set_xlabel('Importancia relativa (%)')
    axes[0].set_title('Importancia por boya — Mejor CF')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].xaxis.set_major_formatter(PercentFormatter())

    bars1 = axes[1].barh(feature_names, imp_mean_norm, color='#4A148C', alpha=0.85)
    _add_bar_labels(axes[1], bars1)
    axes[1].set_xlabel('Importancia relativa (%)')
    axes[1].set_title('Importancia por boya — Promedio CFs')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].xaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(f'Importancia de boyas — CONFETTI ({len(CF)} instancias)',
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}_importancia_boyas_confetti.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_importancia_boyas_confetti.{fig_fmt}")

    plt.show()


def plot_boya_importance_comparison(TSMULE, CF, feature_names=None,
                                     fig_fmt='png', save_path=None):
    """
    Subplot 2x2 comparando importancia por boya entre ts-MULE y CONFETTI.

    Layout:
        (0,0) ts-MULE  — Relevancia media         (#4CAF50 verde vivo)
        (0,1) CONFETTI — Modificación promedio CFs (#9C27B0 púrpura vivo)
        (1,0) ts-MULE  — Relevancia máxima         (#1B5E20 verde oscuro)
        (1,1) CONFETTI — Modificación mejor CF      (#4A148C púrpura oscuro)

    Args:
        TSMULE (list):        Lista de diccionarios retornados por analyze_with_tsmule().
        CF (list):            Lista de diccionarios retornados por analyze_with_confetti().
        feature_names (list): Nombres de las features/boyas. Default: None.
        fig_fmt (str):        Formato de figura. Default: 'png'.
        save_path (str):      Ruta base para guardar figura. Default: None.
    """
    imp_mean_tsmule, imp_max_tsmule, feature_names = compute_boya_importance_tsmule(
        TSMULE, feature_names=feature_names
    )
    imp_best_cf, imp_mean_cf, _ = compute_boya_importance(
        CF, feature_names=feature_names
    )

    def norm(x):
        return x / x.sum() * 100

    config = [
        (0, 0, norm(imp_mean_tsmule), 'steelblue', 'ts-MULE — Relevancia media'),
        (0, 1, norm(imp_mean_cf),     'violet', 'CONFETTI — Modificación promedio CFs'),
        (1, 0, norm(imp_max_tsmule),  'navy', 'ts-MULE — Relevancia máxima'),
        (1, 1, norm(imp_best_cf),     'darkmagenta', 'CONFETTI — Modificación mejor CF'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)

    for row, col, data, color, title in config:
        ax = axes[row, col]
        bars = ax.barh(feature_names, data, color=color, alpha=0.85)
        _add_bar_labels(ax, bars)
        ax.set_xlabel('Importancia relativa (%)')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        ax.xaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(f'Importancia de boyas — ts-MULE vs CONFETTI\n'
                 f'({len(TSMULE)} instancias)', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}_importancia_boyas_comparacion.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_importancia_boyas_comparacion.{fig_fmt}")

    plt.show()


def plot_boya_modification_frequency(CF, feature_names=None, percentil=25,
                                      fig_fmt='png', save_path=None):
    """
    Gráfico de frecuencia de modificación por boya en los contrafactuales.
    Layout 1x2: mejor CF (izq) y promedio CFs (der).
    Valores en porcentaje.

    Args:
        CF (list):            Lista de diccionarios retornados por analyze_with_confetti().
        feature_names (list): Nombres de las features/boyas. Default: None.
        threshold (float):    Umbral para considerar modificación. Default: 1e-6.
                              Usar compute_modification_threshold() para calcularlo.
        fig_fmt (str):        Formato de figura. Default: 'png'.
        save_path (str):      Ruta base para guardar figura. Default: None.
    """
    threshold = compute_modification_threshold(CF, percentil=percentil)
    freq_best, freq_mean, feature_names = compute_boya_modification_frequency(
        CF, feature_names=feature_names, threshold=threshold
    )

    freq_best_pct = freq_best * 100
    freq_mean_pct = freq_mean * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

    bars0 = axes[0].barh(feature_names, freq_best_pct, color='#4A148C', alpha=0.85)
    _add_bar_labels(axes[0], bars0)
    axes[0].set_xlabel('Frecuencia de modificación (%)')
    axes[0].set_title('Frecuencia de modificación — Mejor CF')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_xlim(0, 110)
    axes[0].xaxis.set_major_formatter(PercentFormatter())

    bars1 = axes[1].barh(feature_names, freq_mean_pct, color='#9C27B0', alpha=0.85)
    _add_bar_labels(axes[1], bars1)
    axes[1].set_xlabel('Frecuencia de modificación (%)')
    axes[1].set_title('Frecuencia de modificación — Promedio CFs')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_xlim(0, 110)
    axes[1].xaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(f'Frecuencia de modificación por boya — CONFETTI\n'
                 f'({len(CF)} instancias)', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}_frecuencia_modificacion_boyas.{fig_fmt}',
                    format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_frecuencia_modificacion_boyas.{fig_fmt}")

    plt.show()
