# xai_utils - Utilidades para análisis XAI de series de tiempo
# Módulos disponibles:
#   - tsmule_analysis:   función analyze_with_tsmule
#   - tsmule_plots:      funciones de graficación ts-MULE
#   - confetti_analysis: función analyze_with_confetti
#   - confetti_plots:    funciones de graficación CONFETTI
#   - comparison_plots:  funciones de graficación comparativa

from .tsmule_analysis import analyze_with_tsmule
from .tsmule_plots import (
    plot_relevance_map,
    plot_original_vs_perturbed,
    plot_relevance_per_feature,
    plot_relevance_summary
)
from .confetti_analysis import analyze_with_confetti
from .confetti_plots import (
    plot_best_counterfactual,
    plot_counterfactuals_by_diff,
    plot_counterfactuals_by_ts,
    plot_counterfactual_channels,
    plot_cf_summary
)
from .comparison_plots import (
    plot_comparison_summary,
    plot_comparison_regions,
    plot_comparison_heatmaps
)

__all__ = [
    # ts-MULE
    'analyze_with_tsmule',
    'plot_relevance_map',
    'plot_original_vs_perturbed',
    'plot_relevance_per_feature',
    'plot_relevance_summary',
    # CONFETTI
    'analyze_with_confetti',
    'plot_best_counterfactual',
    'plot_counterfactuals_by_diff',
    'plot_counterfactuals_by_ts',
    'plot_counterfactual_channels',
    'plot_cf_summary',
    # Comparación
    'plot_comparison_summary',
    'plot_comparison_regions',
    'plot_comparison_heatmaps'
]
from .confetti_convergence import run_confetti_convergence

__all__ += ['run_confetti_convergence']
from .tsmule_convergence import run_tsmule_convergence

__all__ += ['run_tsmule_convergence']
