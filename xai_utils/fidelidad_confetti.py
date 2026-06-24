"""
xai_utils/fidelidad_confetti.py
================================
Funciones de fidelidad para evaluación de contrafactuales generados con CONFETTI.

Métricas implementadas (Molnar, 2023; Cetina et al., 2025):
    1. Cambio de clase  — el CF efectivamente cambia la clase predicha
    2. Sparsity         — CONFETTI genera más de un CF (diversidad de soluciones)
    3. Proximity        — el CF modifica la señal menos que el NUN (referencia mínima)
    4. Plausibility     — el espectro del CF es consistente con señales reales de X_train

Nota sobre plausibilidad:
    Se descartó la métrica kNN euclidiana normalizada por ser inadecuada para
    series de tiempo en alta dimensión. Se adoptó un análisis espectral basado
    en similitud coseno entre el espectro del CF y la distribución de referencia
    de X_train, calculada sobre la misma ventana temporal que el CF para garantizar
    comparabilidad frecuencial sin interpolación.

Referencias:
    Molnar, C. (2023). Interpretable Machine Learning (3rd ed.).
    Cetina et al. (2025). Counterfactual Explainable AI Method for Deep
        Learning-Based Multivariate Time Series Classification. AAAI.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


# =============================================================================
# 1. UTILIDADES DE VENTANA
# =============================================================================

def expandir_ventana(t_ini, t_fin, n_timesteps_total, min_ventana=61):
    """
    Expande la ventana modificada por CONFETTI hasta min_ventana timesteps,
    centrada en la zona modificada original. Si choca con un borde,
    compensa expandiendo hacia el lado contrario.

    Si la ventana modificada ya es mayor que min_ventana, se usa tal cual.

    Parámetros
    ----------
    t_ini : int
        Primer timestep modificado por CONFETTI.
    t_fin : int
        Último timestep modificado por CONFETTI.
    n_timesteps_total : int
        Longitud total de la serie (241 en este proyecto).
    min_ventana : int
        Tamaño mínimo de ventana en timesteps. Default=61 (≥1 período
        dominante de tsunami, ~34-40 min, con margen hasta 60 min).

    Retorna
    -------
    t_ini_exp, t_fin_exp : int
        Índices de la ventana expandida.
    """
    largo_actual = t_fin - t_ini + 1
    if largo_actual >= min_ventana:
        return t_ini, t_fin

    expansion     = min_ventana - largo_actual
    expansion_izq = expansion // 2
    expansion_der = expansion - expansion_izq

    t_ini_exp = max(0, t_ini - expansion_izq)
    t_fin_exp = min(n_timesteps_total - 1, t_fin + expansion_der)

    # Compensar si chocamos con algún borde
    if t_ini_exp == 0:
        t_fin_exp = min(n_timesteps_total - 1, min_ventana - 1)
    if t_fin_exp == n_timesteps_total - 1:
        t_ini_exp = max(0, n_timesteps_total - min_ventana)

    return t_ini_exp, t_fin_exp


def detectar_zona_modificada(original, cf, umbral=1e-6):
    """
    Detecta los timesteps donde el CF difiere del original.

    Retorna
    -------
    timesteps_mod : np.ndarray
        Índices de timesteps modificados.
    t_ini, t_fin : int o None
        Primer y último timestep modificado.
    """
    diff_mask    = np.abs(cf - original).mean(axis=1) > umbral
    timesteps_mod = np.where(diff_mask)[0]
    t_ini = int(timesteps_mod[0])  if len(timesteps_mod) > 0 else None
    t_fin = int(timesteps_mod[-1]) if len(timesteps_mod) > 0 else None
    return timesteps_mod, t_ini, t_fin


# =============================================================================
# 2. ESPECTRO DE REFERENCIA
# =============================================================================

def calcular_espectro_referencia(X_train, t_ini_ref, t_fin_ref):
    """
    Calcula la distribución espectral de referencia sobre X_train,
    usando la ventana temporal [t_ini_ref, t_fin_ref].

    Las series se normalizan por amplitud máxima antes de la FFT para
    comparar exclusivamente la estructura frecuencial, independiente
    de la amplitud (que varía significativamente entre clases).

    Parámetros
    ----------
    X_train : np.ndarray, shape (n, timesteps, channels)
    t_ini_ref : int
    t_fin_ref : int

    Retorna
    -------
    freqs : np.ndarray
        Frecuencias en ciclos/minuto (asumiendo dt=1 min).
    esp_median : np.ndarray
        Mediana espectral sobre instancias y canales.
    esp_p5 : np.ndarray
        Percentil 5 de la distribución espectral.
    esp_p95 : np.ndarray
        Percentil 95 de la distribución espectral.
    esp_todas : np.ndarray, shape (n, n_freqs)
        Espectros individuales promediados sobre canales.
    umbral_p5 : float
        Similitud coseno mínima aceptable (percentil 5 de similitudes
        intra-referencia). Umbral de plausibilidad.
    """
    from numpy.linalg import norm

    n_instancias  = X_train.shape[0]
    n_canales     = X_train.shape[2]
    largo_ventana = t_fin_ref - t_ini_ref + 1

    freqs   = np.fft.rfftfreq(largo_ventana, d=1.0)
    n_freqs = len(freqs)

    espectros = np.zeros((n_instancias, n_freqs, n_canales))
    for i in range(n_instancias):
        for ch in range(n_canales):
            serie    = X_train[i, t_ini_ref:t_fin_ref + 1, ch]
            amp_max  = np.abs(serie).max()
            serie_norm = serie / amp_max if amp_max > 1e-10 else serie
            espectros[i, :, ch] = np.abs(np.fft.rfft(serie_norm))

    # Promedio sobre canales → distribución sobre instancias
    esp_todas  = espectros.mean(axis=2)          # (n, n_freqs)
    esp_median = np.median(esp_todas, axis=0)
    esp_p5     = np.percentile(esp_todas,  5, axis=0)
    esp_p95    = np.percentile(esp_todas, 95, axis=0)

    # Umbral P5: similitud coseno mínima observada en el 95% de instancias reales
    sims_ref = np.array([
        np.dot(esp_todas[i], esp_median) /
        (norm(esp_todas[i]) * norm(esp_median) + 1e-10)
        for i in range(n_instancias)
    ])
    umbral_p5 = float(np.percentile(sims_ref, 5))

    return freqs, esp_median, esp_p5, esp_p95, esp_todas, umbral_p5


# =============================================================================
# 3. PLAUSIBILIDAD ESPECTRAL
# =============================================================================

def plausibilidad_espectral(cf, t_ini, t_fin, X_train,
                             n_timesteps_total=241, min_ventana=61):
    """
    Calcula la plausibilidad espectral del CF comparando su espectro
    normalizado contra la distribución de referencia de X_train.

    El espectro de referencia se calcula sobre la misma ventana temporal
    que el CF, garantizando comparabilidad frecuencial sin interpolación.

    Parámetros
    ----------
    cf : np.ndarray, shape (timesteps, channels)
    t_ini : int
        Primer timestep modificado por CONFETTI.
    t_fin : int
        Último timestep modificado por CONFETTI.
    X_train : np.ndarray, shape (n, timesteps, channels)
        Datos de entrenamiento — referencia espectral.
    n_timesteps_total : int
    min_ventana : int

    Retorna
    -------
    similitud : float
        Similitud coseno CF vs mediana de referencia [0, 1].
    percentil : float
        Percentil de la similitud del CF en la distribución intra-referencia.
    es_plausible : bool
        True si similitud >= umbral P5 de referencia.
    t_ini_exp, t_fin_exp : int
        Ventana efectivamente usada.
    freqs : np.ndarray
        Frecuencias del espectro calculado.
    esp_median : np.ndarray
        Mediana de referencia (para graficar).
    esp_p5, esp_p95 : np.ndarray
        Banda de referencia (para graficar).
    umbral_p5 : float
        Umbral de plausibilidad usado.
    """
    from numpy.linalg import norm

    # Expandir ventana si es necesario
    t_ini_exp, t_fin_exp = expandir_ventana(t_ini, t_fin,
                                             n_timesteps_total, min_ventana)

    # Calcular espectro de referencia para esta ventana específica
    freqs, esp_median, esp_p5, esp_p95, esp_todas, umbral_p5 = \
        calcular_espectro_referencia(X_train, t_ini_exp, t_fin_exp)

    # Espectro del CF sobre la misma ventana, promediado sobre canales
    n_canales = cf.shape[1]
    esp_cf_canales = np.zeros((len(freqs), n_canales))
    for ch in range(n_canales):
        serie      = cf[t_ini_exp:t_fin_exp + 1, ch]
        amp_max    = np.abs(serie).max()
        serie_norm = serie / amp_max if amp_max > 1e-10 else serie
        esp_cf_canales[:, ch] = np.abs(np.fft.rfft(serie_norm))
    esp_cf = esp_cf_canales.mean(axis=1)

    # Similitud coseno CF vs mediana de referencia
    similitud = float(np.dot(esp_cf, esp_median) /
                      (norm(esp_cf) * norm(esp_median) + 1e-10))

    # Percentil en distribución intra-referencia
    sims_ref = np.array([
        np.dot(esp_todas[i], esp_median) /
        (norm(esp_todas[i]) * norm(esp_median) + 1e-10)
        for i in range(len(esp_todas))
    ])
    percentil    = float(percentileofscore(sims_ref, similitud))
    es_plausible = similitud >= umbral_p5

    return (similitud, percentil, es_plausible,
            t_ini_exp, t_fin_exp,
            freqs, esp_median, esp_p5, esp_p95, umbral_p5)


# =============================================================================
# 4. VISUALIZACIÓN
# =============================================================================

def plot_cf_channels(original, cf, original_label, cf_label, channel_names=None):
    """
    Grafica la instancia original vs el CF canal por canal,
    resaltando la zona modificada.

    Retorna
    -------
    t_ini, t_fin : int o None
    n_mod : int
    """
    n_channels = original.shape[1]
    if channel_names is None:
        channel_names = [f'Boya {i + 1}' for i in range(n_channels)]

    timesteps_mod, t_ini, t_fin = detectar_zona_modificada(original, cf)
    n_mod = len(timesteps_mod)

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(original[:, i], color='blue',
                label=f'Original (label={original_label})')
        ax.plot(cf[:, i], color='red', linestyle='--',
                label=f'CF (label={cf_label})')
        if t_ini is not None:
            ax.axvspan(t_ini, t_fin, alpha=0.2, color='yellow',
                       label=f'Zona modificada: t={t_ini}-{t_fin}')
        ax.set_ylabel(channel_names[i])
        ax.legend(fontsize=7)

    plt.xlabel('Timestep (minutos)')
    plt.suptitle(
        f'Original (label={original_label}) → Contrafactual (label={cf_label})\n'
        f'{n_mod} timesteps modificados'
        + (f' (t={t_ini}-{t_fin})' if t_ini is not None else ''),
        fontsize=11
    )
    plt.tight_layout()
    plt.show()
    return t_ini, t_fin, n_mod


def plot_plausibilidad_espectral(cf, original, original_label, cf_label,
                                  t_ini, t_fin, t_ini_exp, t_fin_exp,
                                  freqs, esp_median, esp_p5, esp_p95,
                                  similitud, percentil, es_plausible):
    """
    Gráfico de dos paneles:
    - Panel izquierdo: señal original vs CF en la ventana analizada,
      con zona modificada por CONFETTI resaltada.
    - Panel derecho: espectro del CF vs banda P5-P95 de referencia
      y mediana de referencia, ambos calculados sobre la misma ventana.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Panel izquierdo: señal (promedio de canales) ---
    ax        = axes[0]
    t_ventana = np.arange(t_ini_exp, t_fin_exp + 1)
    orig_mean = original[t_ini_exp:t_fin_exp + 1, :].mean(axis=1)
    cf_mean   = cf[t_ini_exp:t_fin_exp + 1, :].mean(axis=1)

    ax.plot(t_ventana, orig_mean, color='blue',
            label=f'Original (label={original_label})')
    ax.plot(t_ventana, cf_mean, color='red', linestyle='--',
            label=f'CF (label={cf_label})')
    if t_ini is not None:
        ax.axvspan(t_ini, t_fin, alpha=0.2, color='yellow',
                   label=f'Zona modificada: t={t_ini}-{t_fin}')
    ax.set_xlabel('Timestep (minutos)')
    ax.set_ylabel('Amplitud media')
    ax.set_title(f'Señal original vs CF\n(ventana analizada: t={t_ini_exp}-{t_fin_exp})')
    ax.legend(fontsize=8)

    # --- Panel derecho: espectro ---
    ax = axes[1]
    # Excluir componente DC (freq=0) para el gráfico
    freqs_plot = freqs[1:]
    periodos   = 1.0 / freqs_plot

    ax.fill_between(periodos, esp_p5[1:], esp_p95[1:],
                    alpha=0.3, color='blue',
                    label='Banda P5-P95 referencia (X_train)')
    ax.plot(periodos, esp_median[1:], color='blue', linewidth=1.5,
            linestyle='--', label='Mediana referencia')

    # Espectro del CF
    esp_cf_plot = np.zeros(len(freqs))
    for ch in range(cf.shape[1]):
        serie      = cf[t_ini_exp:t_fin_exp + 1, ch]
        amp_max    = np.abs(serie).max()
        serie_norm = serie / amp_max if amp_max > 1e-10 else serie
        esp_cf_plot += np.abs(np.fft.rfft(serie_norm))
    esp_cf_plot /= cf.shape[1]

    veredicto = '✅ Plausible' if es_plausible else '❌ No plausible'
    ax.plot(periodos, esp_cf_plot[1:], color='red', linewidth=2,
            label=f'CF (sim={similitud:.3f}, P{percentil:.0f}) {veredicto}')

    ax.set_xlabel('Período (minutos)')
    ax.set_ylabel('Magnitud espectral normalizada')
    ax.set_title('Análisis espectral — CF vs referencia X_train\n'
                 f'(ventana: {t_fin_exp - t_ini_exp + 1} timesteps)')
    ax.legend(fontsize=8)
    largo = t_fin_exp - t_ini_exp + 1
    ax.set_xlim([2, largo])

    plt.suptitle(
        f'Plausibilidad espectral — label {original_label}→{cf_label}',
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. REPORTE DEL MEJOR CF
# =============================================================================

def reporte_cf(original, original_label, cf, cf_label, all_cfs, nun,
               X_train, n_timesteps_total=241, min_ventana=61):
    """
    Evalúa y reporta los 4 criterios de calidad del mejor CF según Molnar (2023).

    El espectro de referencia se calcula sobre la misma ventana efectiva del CF,
    garantizando comparabilidad frecuencial sin interpolación.

    Parámetros
    ----------
    original : np.ndarray, shape (timesteps, channels)
    original_label : int
    cf : np.ndarray, shape (timesteps, channels)
    cf_label : int
    all_cfs : list of Counterfactual
    nun : np.ndarray, shape (timesteps, channels)
    X_train : np.ndarray
    n_timesteps_total : int
    min_ventana : int

    Retorna
    -------
    dict con todos los valores calculados.
    """
    # ---------------------------------------------------------------
    # 1. CAMBIO DE CLASE
    # ---------------------------------------------------------------
    cambio_exitoso = (cf_label != original_label)

    # ---------------------------------------------------------------
    # 2. SPARSITY
    # ---------------------------------------------------------------
    n_cfs_total    = len(all_cfs)
    cumple_sparsity = n_cfs_total > 1

    # ---------------------------------------------------------------
    # 3. PROXIMITY
    # ---------------------------------------------------------------
    timesteps_mod, t_ini, t_fin = detectar_zona_modificada(original, cf)
    n_timesteps_mod = len(timesteps_mod)
    pct_mod         = 100 * n_timesteps_mod / original.shape[0]

    diff_nun_mask   = np.abs(nun - original).mean(axis=1) > 1e-6
    n_timesteps_nun = int(np.sum(diff_nun_mask))
    pct_nun         = 100 * n_timesteps_nun / original.shape[0]

    energia_original  = np.sqrt(np.sum(original ** 2))
    dist_cf           = np.sqrt(np.sum((cf - original) ** 2))
    dist_nun          = np.sqrt(np.sum((nun - original) ** 2))
    prox_cf_relativa  = dist_cf / energia_original
    prox_nun_relativa = dist_nun / energia_original
    proximity_ratio   = dist_cf / dist_nun
    cumple_proximity  = proximity_ratio <= 1

    # ---------------------------------------------------------------
    # 4. PLAUSIBILITY ESPECTRAL
    # ---------------------------------------------------------------
    if t_ini is None:
        similitud, percentil, es_plausible = 0.0, 0.0, False
        t_ini_exp = t_fin_exp = 0
        freqs = esp_median = esp_p5 = esp_p95 = None
        umbral_p5 = 0.0
    else:
        (similitud, percentil, es_plausible,
         t_ini_exp, t_fin_exp,
         freqs, esp_median, esp_p5, esp_p95, umbral_p5) = plausibilidad_espectral(
            cf=cf, t_ini=t_ini, t_fin=t_fin,
            X_train=X_train,
            n_timesteps_total=n_timesteps_total,
            min_ventana=min_ventana
        )

    # ---------------------------------------------------------------
    # REPORTE
    # ---------------------------------------------------------------
    sep = '=' * 58
    print(sep)
    print('        EVALUACIÓN DEL CONTRAFACTUAL')
    print(sep)

    print(f'\n1. CAMBIO DE CLASE')
    print(f'   Label original:            {original_label}')
    print(f'   Label contrafactual:       {cf_label}')
    print(f'   Cumple: {"✅ Sí" if cambio_exitoso else "❌ No"}')

    print(f'\n2. SPARSITY')
    print(f'   CFs generados por CONFETTI:      {n_cfs_total}')
    print(f'   Cumple (>1 CF encontrado):       {"✅ Sí" if cumple_sparsity else "❌ No"}')

    print(f'\n3. PROXIMITY')
    print(f'   Cambio CF   (L2 / energía original): {prox_cf_relativa * 100:.2f}%')
    print(f'   Cambio NUN  (L2 / energía original): {prox_nun_relativa * 100:.2f}%  [referencia]')
    print(f'   Ratio CF/NUN:                        {proximity_ratio:.4f}')
    print(f'   Timesteps modificados (CF):  {n_timesteps_mod} / {original.shape[0]} ({pct_mod:.1f}%)')
    print(f'   Timesteps distintos NUN:     {n_timesteps_nun} / {original.shape[0]} ({pct_nun:.1f}%) [referencia]')
    print(f'   Cumple (ratio ≤ 1):          {"✅ Sí" if cumple_proximity else "❌ No"}')

    print(f'\n4. PLAUSIBILITY ESPECTRAL')
    print(f'   Ventana analizada: t={t_ini_exp}-{t_fin_exp} ({t_fin_exp - t_ini_exp + 1} timesteps)')
    print(f'   Similitud coseno vs referencia:  {similitud:.4f}')
    print(f'   Percentil en distribución ref.:  {percentil:.1f}%')
    print(f'   Umbral mínimo (P5 referencia):   {umbral_p5:.4f}')
    print(f'   Cumple (similitud ≥ umbral):     {"✅ Sí" if es_plausible else "❌ No"}')

    print(f'\n{sep}')
    criterios = [cambio_exitoso, cumple_sparsity, cumple_proximity, es_plausible]
    nombres   = ['Cambio de clase', 'Sparsity', 'Proximity', 'Plausibility espectral']
    print(f'   RESUMEN: {sum(criterios)}/4 criterios cumplidos')
    for nombre, cumple in zip(nombres, criterios):
        print(f'   {"✅" if cumple else "❌"} {nombre}')
    print(sep)

    return {
        'cambio_exitoso':    cambio_exitoso,
        'n_cfs_total':       n_cfs_total,
        'cumple_sparsity':   cumple_sparsity,
        'n_timesteps_mod':   n_timesteps_mod,
        'n_timesteps_nun':   n_timesteps_nun,
        'pct_mod':           pct_mod,
        't_ini':             t_ini,  't_fin':     t_fin,
        't_ini_exp':         t_ini_exp, 't_fin_exp': t_fin_exp,
        'dist_cf':           dist_cf,   'dist_nun':  dist_nun,
        'prox_cf_relativa':  prox_cf_relativa,
        'prox_nun_relativa': prox_nun_relativa,
        'proximity_ratio':   proximity_ratio,
        'cumple_proximity':  cumple_proximity,
        'similitud_espectral': similitud,
        'percentil_espectral': percentil,
        'umbral_p5':         umbral_p5,
        'es_plausible':      es_plausible,
        'freqs':             freqs,
        'esp_median':        esp_median,
        'esp_p5':            esp_p5,
        'esp_p95':           esp_p95,
    }


# =============================================================================
# 6. EVALUACIÓN DE TODOS LOS CFs
# =============================================================================

def evaluar_todos_cfs(all_cfs, original, original_label, nun, model_wrapped,
                       X_train, n_timesteps_total=241, min_ventana=61):
    """
    Evalúa los 4 criterios para cada CF generado por CONFETTI.
    El espectro de referencia se recalcula para cada CF según su ventana efectiva.
    Retorna la lista ordenada por proximity_ratio (menor cambio primero).

    Parámetros
    ----------
    all_cfs : list of Counterfactual
    original : np.ndarray, shape (timesteps, channels)
    original_label : int
    nun : np.ndarray, shape (timesteps, channels)
    model_wrapped : keras.Model
    X_train : np.ndarray
    n_timesteps_total : int
    min_ventana : int

    Retorna
    -------
    list of dict, ordenada por proximity_ratio ascendente.
    """
    dist_nun = np.sqrt(np.sum((nun - original) ** 2))
    resultados = []

    for i, cf_obj in enumerate(all_cfs):
        cf_i = cf_obj.counterfactual

        # 1. Cambio de clase — verificado con el modelo
        pred         = model_wrapped.predict(cf_i[np.newaxis], verbose=0)
        cf_label_i   = int(np.argmax(pred, axis=1)[0])
        cambio_exitoso = (cf_label_i != original_label)

        # 3. Proximity
        timesteps_mod, t_ini_i, t_fin_i = detectar_zona_modificada(original, cf_i)
        n_timesteps_mod_i = len(timesteps_mod)
        dist_cf_i         = np.sqrt(np.sum((cf_i - original) ** 2))
        proximity_ratio_i = dist_cf_i / dist_nun
        cumple_proximity_i = proximity_ratio_i <= 1

        # 4. Plausibility espectral — referencia calculada para esta ventana
        if t_ini_i is None:
            similitud_i = percentil_i = 0.0
            es_plausible_i = False
            t_ini_exp_i = t_fin_exp_i = 0
        else:
            (similitud_i, percentil_i, es_plausible_i,
             t_ini_exp_i, t_fin_exp_i, *_) = plausibilidad_espectral(
                cf=cf_i, t_ini=t_ini_i, t_fin=t_fin_i,
                X_train=X_train,
                n_timesteps_total=n_timesteps_total,
                min_ventana=min_ventana
            )

        # ¿Cumple los 4 criterios? (sparsity se evalúa globalmente)
        cumple_4 = cambio_exitoso and cumple_proximity_i and es_plausible_i

        resultados.append({
            'idx':              i,
            'cf':               cf_i,
            'cf_label':         cf_label_i,
            'cambio_exitoso':   cambio_exitoso,
            'n_timesteps_mod':  n_timesteps_mod_i,
            't_ini':            t_ini_i,    't_fin':     t_fin_i,
            't_ini_exp':        t_ini_exp_i, 't_fin_exp': t_fin_exp_i,
            'dist_cf':          dist_cf_i,
            'proximity_ratio':  proximity_ratio_i,
            'cumple_proximity': cumple_proximity_i,
            'similitud_espectral': similitud_i,
            'percentil_espectral': percentil_i,
            'es_plausible':     es_plausible_i,
            'cumple_4':         cumple_4,
        })

    return sorted(resultados, key=lambda x: x['proximity_ratio'])


# =============================================================================
# 7. SELECCIÓN DEL MEJOR CF
# =============================================================================

def search_best_cf(resultados_todos, criterio='proximity_ratio'):
    """
    Entre todos los CFs que cumplen los 4 criterios (incluyendo plausibilidad
    espectral), selecciona el mejor según el criterio especificado.

    Siguiendo Molnar (2023), el criterio principal es el menor cambio respecto
    a la instancia original (proximity_ratio).

    Parámetros
    ----------
    resultados_todos : list of dict
        Output de evaluar_todos_cfs().
    criterio : str
        - 'proximity_ratio'      : menor cambio respecto al NUN (default)
        - 'similitud_espectral'  : mayor similitud espectral
        - 'n_timesteps_mod'      : menor número de timesteps modificados

    Retorna
    -------
    dict o None
        El mejor CF según el criterio, o None si ninguno cumple los 4 criterios.
    """
    validos = [r for r in resultados_todos if r['cumple_4']]
    if not validos:
        return None
    reverse = (criterio == 'similitud_espectral')
    return sorted(validos, key=lambda x: x[criterio], reverse=reverse)[0]
