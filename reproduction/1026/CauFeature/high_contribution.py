import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

def select_high_contribution_features(feature_names, contribution_values):
    values = np.asarray(contribution_values, dtype=float)
    if len(values) == 0:
        return [], float("inf"), "empty"
    if len(values) < 10:
        threshold = float(np.percentile(values, 80))
        reason = "small_feature_count_p80"
    else:
        try:
            kde = gaussian_kde(values, bw_method="silverman")
            ordered = np.sort(values)
            density = gaussian_filter1d(kde(ordered), sigma=1)
            second = np.gradient(np.gradient(density))
            points = np.where(np.diff(np.sign(second)) != 0)[0]
            right = [int(i) for i in points if i > len(ordered) * 0.3]
            if right:
                raw = float(ordered[right[-1]])
                threshold = min(raw, float(np.max(values)) * 0.9)
                reason = "right_density_inflection"
            else:
                threshold = float(np.percentile(values, 75))
                reason = "fallback_p75"
        except Exception:
            threshold = float(np.percentile(values, 75))
            reason = "kde_failure_fallback_p75"
    selected = [name for name, value in zip(feature_names, values) if float(value) >= threshold]
    return selected, threshold, reason
