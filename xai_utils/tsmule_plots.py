"""
tsmule_plots.py
---------------
Funciones de graficación para visualizar resultados del análisis ts-MULE.

Funciones disponibles:
    - plot_relevance_map:          Mapa de calor de relevancia promedio
    - plot_original_vs_perturbed:  Serie original vs perturbada con regiones relevantes
    - plot_relevance_per_feature:  Relevancia superpuesta a la serie por boya
    - plot_relevance_summary:      Resumen de relevancia absoluta por boya (barras)

Uso típico:
    from xai_utils import (plot_relevance_map, plot_original_vs_perturbed,
                           plot_relevance_per_feature, plot_relevance_summary)

    feature_names = ['Boya 1', ..., 'Boya 6']
    plot_relevance_map(resultados, feature_names=feature_names, n_runs=100)
    plot_original_vs_perturbed(resultados, feature_names=feature_names, n_runs=100)
    plot_relevance_per_feature(resultados, feature_names=feature_names, n_runs=100)
    plot_relevance_summary(resultados, feature_names=feature_names, n_runs=100)

Autor: Proyecto XAI para series de tiempo de tsunami
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_relevance_map(resultados, feature_names=None, n_runs=None,
                       save_path=None, fig_fmt='png'):
    """
    Mapa de calor de relevancia promedio (timesteps x boyas).

    Args:
        resultados (dict): Diccionario retornado por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.
        n_runs (int): Número de iteraciones usado (para el título). Default: None.
        save_path (str): Ruta base para guardar figura en PDF. Default: None.
    """
    relevance_promedio = resultados['relevance_promedio']
    n_steps, n_features = relevance_promedio.shape

    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(relevance_promedio.T, aspect='auto')

    title = 'Relevancia promedio — ts-MULE'
    if n_runs:
        title += f' (n_runs={n_runs})'
    ax.set_title(title)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Boyas')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names)
    fig.colorbar(im, ax=ax, label='Relevancia')
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}mapa_relevancia.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_mapa_relevancia.pdf")

    plt.show()


def plot_original_vs_perturbed(resultados, feature_names=None, n_runs=None,
                                percentile=90, save_path=None, fig_fmt='png'):
    """
    Serie original vs perturbada con regiones relevantes sombreadas en rojo.

    Args:
        resultados (dict): Diccionario retornado por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.
        n_runs (int): Número de iteraciones usado (para el título). Default: None.
        percentile (float): Percentil usado en la perturbación. Default: 90.
        save_path (str): Ruta base para guardar figura en PDF. Default: None.
    """
    x = resultados['x_original']
    x_perturbed = resultados['x_perturbed']
    n_steps, n_features = x.shape

    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features * 3))

    for i in range(n_features):
        axes[i].plot(x[:, i], label='Serie original',
                     color='steelblue', linewidth=1.5)
        axes[i].plot(x_perturbed[:, i], label=f'Serie perturbada (p{percentile})',
                     color='orange', linewidth=1.5, linestyle='--')

        diff = np.abs(x[:, i] - x_perturbed[:, i]) > 1e-10
        axes[i].fill_between(range(n_steps),
                             axes[i].get_ylim()[0], axes[i].get_ylim()[1],
                             where=diff, alpha=0.2, color='red',
                             label='Segmentos relevantes')
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel('Timesteps')
        axes[i].set_ylabel('Amplitud')
        axes[i].legend(loc='upper right')

    title = f'Serie original vs perturbada — ts-MULE'
    if n_runs:
        title += f' (n_runs={n_runs})'
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}series_perturbadas.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_series_perturbadas.{fig_fmt}")

    plt.show()


def plot_relevance_per_feature(resultados, feature_names=None, n_runs=None,
                                save_path=None, fig_fmt='png'):
    """
    Relevancia promedio superpuesta a la serie original por boya (eje doble).

    Args:
        resultados (dict): Diccionario retornado por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.
        n_runs (int): Número de iteraciones usado (para el título). Default: None.
        save_path (str): Ruta base para guardar figura en PDF. Default: None.
    """
    relevance_promedio = resultados['relevance_promedio']
    x = resultados['x_original']
    n_steps, n_features = x.shape

    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features * 3))

    for i in range(n_features):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.plot(x[:, i], color='steelblue', linewidth=1.5, label='Serie original')
        ax2.plot(relevance_promedio[:, i], color='red', linewidth=1.2,
                 linestyle='--', label='Relevancia')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

        ax1.set_title(feature_names[i])
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Amplitud', color='steelblue')
        ax2.set_ylabel('Relevancia', color='red')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    title = 'Serie original y relevancia por boya — ts-MULE'
    if n_runs:
        title += f' (n_runs={n_runs})'
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}relevancia_por_boya.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_relevancia_por_boya.{fig_fmt}")

    plt.show()


def plot_relevance_summary(resultados, feature_names=None, n_runs=None,
                           save_path=None, fig_fmt='png'):
    """
    Resumen de relevancia absoluta media y máxima por boya (gráfico de barras).
    Útil para comparar importancia relativa entre boyas y con otros métodos XAI.

    Args:
        resultados (dict): Diccionario retornado por analyze_with_tsmule().
        feature_names (list): Nombres de las features/boyas. Default: None.
        n_runs (int): Número de iteraciones usado (para el título). Default: None.
        save_path (str): Ruta base para guardar figura en PDF. Default: None.
    """
    relevance_promedio = resultados['relevance_promedio']
    n_features = relevance_promedio.shape[1]

    if feature_names is None:
        feature_names = [f'Boya {i+1}' for i in range(n_features)]

    rel_abs_mean = np.mean(np.abs(relevance_promedio), axis=0)
    rel_abs_max  = np.max(np.abs(relevance_promedio), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].barh(feature_names, rel_abs_mean, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Relevancia absoluta media')
    axes[0].set_title('Relevancia media por boya')
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(feature_names, rel_abs_max, color='orange', alpha=0.8)
    axes[1].set_xlabel('Relevancia absoluta máxima')
    axes[1].set_title('Relevancia máxima por boya')
    axes[1].grid(axis='x', alpha=0.3)

    title = 'Resumen de relevancia por boya — ts-MULE'
    if n_runs:
        title += f' (n_runs={n_runs})'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(f'{save_path}resumen_relevancia.{fig_fmt}', format=fig_fmt, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_path}_resumen_relevancia.{fig_fmt}")

    plt.show()
