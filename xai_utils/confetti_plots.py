"""
confetti_plots.py
-----------------
Funciones de graficación para visualizar resultados del análisis CONFETTI.

Funciones disponibles:
    - plot_best_counterfactual:        Original vs mejor contrafactual por canal
    - plot_counterfactuals_by_diff:    5 contrafactuales representativos por diff_mean
    - plot_counterfactuals_by_ts:      5 contrafactuales representativos por extensión
    - plot_counterfactual_channels:    Un contrafactual específico, canal por canal
    - plot_cf_summary:                 Resumen estadístico de todos los contrafactuales

Uso típico:
    from xai_utils import (plot_best_counterfactual, plot_counterfactuals_by_diff,
                           plot_counterfactuals_by_ts, plot_counterfactual_channels,
                           plot_cf_summary)

    plot_best_counterfactual(resultados)
    plot_counterfactuals_by_diff(resultados)
    plot_counterfactuals_by_ts(resultados)
    plot_counterfactual_channels(resultados, cf_index=0)
    plot_cf_summary(resultados)

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_best_counterfactual(resultados, save_path=None, fig_fmt='png'):
    """
    Grafica el mejor contrafactual vs original, canal por canal.

    Args:
        resultados (dict): Diccionario retornado por analyze_with_confetti().
        save_path (str):   Ruta base para guardar figura en PDF. Default: None.
    """
    instance = resultados['instance']
    best_cf = resultados['best_cf']
    feature_names = resultados['feature_names']
    n_features = instance.shape[1]

    diff = np.abs(best_cf - instance)
    mask = diff.mean(axis=1) > 1e-6
    t_inicio = int(np.where(mask)[0][0])  if np.any(mask) else 0
    t_fin    = int(np.where(mask)[0][-1]) if np.any(mask) else instance.shape[0]

    fig, axes = plt.subplots(n_features, 1, figsize=(14, n_features * 2.5), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(instance[:, i], label='Original (inunda)', color='steelblue', linewidth=1.5)
        ax.plot(best_cf[:, i], label='Contrafactual (no inunda)',
                color='red', linewidth=1.5, linestyle='--')
        ax.axvspan(t_inicio, t_fin, alpha=0.2, color='yellow',
                   label=f'Zona modificada (t={t_inicio}-{t_fin})')
        ax.set_ylabel(feature_names[i])
        ax.legend(fontsize=7)

    plt.xlabel('Timestep')
    plt.suptitle('Original vs Mejor contrafactual — CONFETTI', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}mejor_cf.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_mejor_cf.{fig_fmt}")

    plt.show()


def plot_counterfactuals_by_diff(resultados, save_path=None, fig_fmt='png'):
    """
    Grafica 5 contrafactuales representativos ordenados por diff_mean (promedio sobre canales).

    Args:
        resultados (dict): Diccionario retornado por analyze_with_confetti().
        save_path (str):   Ruta base para guardar figura en PDF. Default: None.
    """
    instance = resultados['instance']
    cf_stats_sorted = resultados['cf_stats_sorted_diff']
    n = len(cf_stats_sorted)

    indices_rep = [0, n//4, n//2, 3*n//4, n-1]
    labels_rep  = ['Mínima', 'Baja', 'Media', 'Alta', 'Máxima']

    fig, axes = plt.subplots(len(indices_rep), 1,
                             figsize=(14, 3 * len(indices_rep)), sharex=True)

    for ax, idx, lbl in zip(axes, indices_rep, labels_rep):
        s = cf_stats_sorted[idx]
        orig_mean = instance.mean(axis=1)
        cf_mean   = s['cf'].mean(axis=1)

        ax.plot(orig_mean, label='Original', color='steelblue')
        ax.plot(cf_mean, label=f'Contrafactual ({lbl})', color='red', linestyle='--')
        if s['t_inicio'] is not None:
            ax.axvspan(s['t_inicio'], s['t_fin'], alpha=0.2, color='yellow',
                       label='Zona modificada')
        ax.set_title(f'Modificación {lbl} — Δmean={s["diff_mean"]:.5f}, '
                     f'timesteps={s["timesteps_modificados"]}')
        ax.legend(fontsize=8)
        ax.set_ylabel('Amplitud media')

    plt.xlabel('Timestep')
    plt.suptitle('Contrafactuales ordenados por magnitud de modificación — CONFETTI',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}cf_por_diff.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_cf_por_diff.{fig_fmt}")

    plt.show()


def plot_counterfactuals_by_ts(resultados, save_path=None, fig_fmt='png'):
    """
    Grafica 5 contrafactuales representativos ordenados por extensión temporal modificada.

    Args:
        resultados (dict): Diccionario retornado por analyze_with_confetti().
        save_path (str):   Ruta base para guardar figura en PDF. Default: None.
    """
    instance = resultados['instance']
    cf_stats_sorted = resultados['cf_stats_sorted_ts']
    n = len(cf_stats_sorted)

    indices_rep = [0, n//4, n//2, 3*n//4, n-1]
    labels_rep  = ['Mínima extensión', 'Baja', 'Media', 'Alta', 'Máxima extensión']

    fig, axes = plt.subplots(len(indices_rep), 1,
                             figsize=(14, 3 * len(indices_rep)), sharex=True)

    for ax, idx, lbl in zip(axes, indices_rep, labels_rep):
        s = cf_stats_sorted[idx]
        orig_mean = instance.mean(axis=1)
        cf_mean   = s['cf'].mean(axis=1)

        ax.plot(orig_mean, label='Original', color='steelblue')
        ax.plot(cf_mean, label=f'Contrafactual ({lbl})', color='red', linestyle='--')
        ax.axvspan(s['t_inicio'], s['t_fin'], alpha=0.2, color='yellow',
                   label=f"t={s['t_inicio']}-{s['t_fin']} ({s['timesteps_modificados']} ts)")
        ax.set_title(f"{lbl} — {s['timesteps_modificados']} timesteps modificados, "
                     f"Δmean={s['diff_mean']:.5f}")
        ax.legend(fontsize=8)
        ax.set_ylabel('Amplitud media')

    plt.xlabel('Timestep')
    plt.suptitle('Contrafactuales ordenados por extensión de modificación — CONFETTI\n'
                 '(comportamiento promedio de la onda a lo largo de las boyas)',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}cf_por_ts.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_cf_por_ts.{fig_fmt}")

    plt.show()


def plot_counterfactual_channels(resultados, cf_index, cf_sorted_key='ts', save_path=None, fig_fmt='png'):
    """
    Grafica un contrafactual específico canal por canal.

    Args:
        resultados (dict):   Diccionario retornado por analyze_with_confetti().
        cf_index (int):      Índice del contrafactual a graficar.
        cf_sorted_key (str): Lista ordenada a usar: 'ts' | 'diff' | 'region'. Default: 'ts'.
        save_path (str):     Ruta base para guardar figura en PDF. Default: None.
    """
    key_map = {
        'ts':     'cf_stats_sorted_ts',
        'diff':   'cf_stats_sorted_diff',
        'region': 'cf_stats_sorted_region'
    }
    cf_stats_sorted = resultados[key_map.get(cf_sorted_key, 'cf_stats_sorted_ts')]
    instance = resultados['instance']
    feature_names = resultados['feature_names']
    n_features = instance.shape[1]

    s = cf_stats_sorted[cf_index]
    cf = s['cf']

    fig, axes = plt.subplots(n_features, 1, figsize=(14, n_features * 2.5), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(instance[:, i], label='Original (inunda)', color='steelblue', linewidth=1.5)
        ax.plot(cf[:, i], label='Contrafactual (no inunda)', color='red',
                linewidth=1.5, linestyle='--')
        if s['t_inicio'] is not None:
            ax.axvspan(s['t_inicio'], s['t_fin'], alpha=0.2, color='yellow',
                       label=f"Zona modificada: t={s['t_inicio']}-{s['t_fin']}")
        ax.set_ylabel(feature_names[i])
        ax.legend(fontsize=7)

    plt.xlabel('Timestep')
    plt.suptitle(
        f"Contrafactual #{cf_index} — "
        f"{s['timesteps_modificados']} timesteps modificados | "
        f"t={s['t_inicio']}-{s['t_fin']} | "
        f"Δmean={s['diff_mean']:.5f}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}cf_canales_{cf_index}.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_cf_canales_{cf_index}.{fig_fmt}")

    plt.show()


def plot_cf_summary(resultados, save_path=None, fig_fmt='png'):
    """
    Resumen de modificación por boya sobre todos los contrafactuales generados.
    Comparable con plot_relevance_summary de ts-MULE:
        - Modificación media por boya  ↔  Relevancia absoluta media
        - Modificación máxima por boya ↔  Relevancia absoluta máxima

    Args:
        resultados (dict): Diccionario retornado por analyze_with_confetti().
        save_path (str):   Ruta base para guardar figura en PDF. Default: None.
    """
    cf_stats      = resultados['cf_stats']
    instance      = resultados['instance']
    feature_names = resultados['feature_names']
    n_features    = instance.shape[1]

    # Calcular modificación por boya sobre todos los contrafactuales
    diffs_por_boya = np.array([
        np.mean(np.abs(s['cf'] - instance), axis=0)
        for s in cf_stats if s['t_inicio'] is not None
    ])  # shape (n_cfs, n_features)

    diff_mean_por_boya = diffs_por_boya.mean(axis=0)  # (n_features,)
    diff_max_por_boya  = diffs_por_boya.max(axis=0)   # (n_features,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].barh(feature_names, diff_mean_por_boya, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Modificación media')
    axes[0].set_title('Modificación media por boya')
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(feature_names, diff_max_por_boya, color='orange', alpha=0.8)
    axes[1].set_xlabel('Modificación máxima')
    axes[1].set_title('Modificación máxima por boya')
    axes[1].grid(axis='x', alpha=0.3)

    fig.suptitle(f'Resumen de modificación por boya — CONFETTI '
                 f'({len(cf_stats)} contrafactuales)', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}resumen_cf.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_resumen_cf.{fig_fmt}")

    plt.show()
