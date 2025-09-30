import os
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import norm, spearmanr, gaussian_kde
from itertools import combinations
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
import shared_globals
from multiprocessing import Pool, Lock
import threading


def get_and_print_high_contrib_features(features, print_lock):
    """提取高贡献特征名称并打印（仅输出名称）"""

    high_contrib_names = [f.name for f in features if f.is_high_contrib]

    with print_lock:
        print(f"\n【High Contribution Feature List】total sum:{len(high_contrib_names)}:")
        for i, name in enumerate(high_contrib_names, 1):
            print(f"{i}. {name} (Contribution Value: {[f.con_i for f in features if f.name == name][0]:.4f})")

    return high_contrib_names


class CausalConfig:
    def __init__(self):

        self.high_contrib_method = "data_driven"
        self.alpha_adjust_strategy = "adaptive"
        self.stability_threshold_method = "fdr"
        self.auto_tune_parameters = True

        self.max_alpha_reduction = 0.5
        self.min_stability_threshold = 0.2
        self.max_cond_reduction = 1
        self.fdr_control_level = 0.2
        self.small_sample_threshold = 10
        self.con_sigma = 0.5
        self.high_contrib_fallback_percentile = 75
        self.n_bootstrap = 30
        self.con_percentile = 30
        self.noise_percentile = 70
        self.min_valid_paths = 1

    def auto_tune_based_on_data(self, features):
        """基于数据特征自动调整参数"""
        if not self.auto_tune_parameters:
            return

        n_features = len(features)
        self.fdr_control_level = max(0.05, min(0.3, 0.2 * (1 + np.log10(n_features / 10))))

        if n_features < 10:
            self.max_path_length = 2
        elif n_features < 50:
            self.max_path_length = 3
        else:
            self.max_path_length = 4

        avg_noise = np.mean([f.noise for f in features])
        self.min_stability_threshold = max(0.1, min(0.3, 0.2 * (1 - avg_noise)))

    def get_parameter_explanations(self):
        """返回所有关键参数的详细说明 - 修正版本"""

        return {

            "high_contrib_method": {
                "Value": self.high_contrib_method,
                "Explanation": "High contribution feature determination method: 'data_driven' means automatically determining the threshold based on the density inflection point of the contribution value distribution to avoid manual proportion setting"
            },
            "alpha_adjust_strategy": {
                "Value": self.alpha_adjust_strategy,
                "Explanation": "Alpha adjustment strategy: 'adaptive' means dynamic adjustment based on feature contribution values, with larger adjustment ranges for higher contribution values"
            },
            "stability_threshold_method": {
                "Value": self.stability_threshold_method,
                "Explanation": "Path stability threshold calculation method: 'fdr' means using the Benjamini-Hochberg method to control the false discovery rate"
            },

            "max_alpha_reduction": {
                "Value": self.max_alpha_reduction,
                "Explanation": "Maximum alpha reduction ratio, controlling the upper limit of independence test threshold adjustment for high-contribution feature pairs"
            },
            "min_stability_threshold": {
                "Value": self.min_stability_threshold,
                "Explanation": "Minimum threshold for path stability; paths exceeding this threshold are retained even if FDR calculation is not significant"
            },
            "max_cond_reduction": {
                "Value": self.max_cond_reduction,
                "Explanation": "Maximum reduction in conditional combinations; the maximum number of conditional combinations reduced for high-contribution feature pairs during skeleton construction"
            },
            "fdr_control_level": {
                "Value": self.fdr_control_level,
                "Explanation": "False Discovery Rate (FDR) control level, controlling the upper limit of false positive proportion in path selection"
            },
            "small_sample_threshold": {
                "Value": self.small_sample_threshold,
                "Explanation": "Small sample determination threshold; when the number of contribution value samples is less than this value, switch to a more conservative high-contribution determination method"
            },
            "con_sigma": {
                "Value": self.con_sigma,
                "Explanation": "Standard deviation of the contribution value difference penalty term, controlling the impact of contribution value differences on the independence test"
            },
            "high_contrib_fallback_percentile": {
                "Value": self.high_contrib_fallback_percentile,
                "Explanation": "Fallback percentile used to determine high-contribution features when no density inflection point can be found"
            }
        }

    def print_parameter_summary(self):
        """打印参数摘要说明"""
        print("\n" + "=" * 50)
        print("Causal Graph Construction Parameter Explanation")
        print("=" * 50)

        explanations = self.get_parameter_explanations()

        print("\n【Method Selection Parameters】")
        print("-" * 40)
        for param in ["high_contrib_method", "alpha_adjust_strategy", "stability_threshold_method"]:
            print(f"{param}:")
            print(f"  Value: {explanations[param]['Value']}")
            print(f"  Explanation: {explanations[param]['Explanation']}\n")

        print("\n【Numerical Parameters】")
        print("-" * 40)
        for param in ["max_alpha_reduction", "min_stability_threshold", "max_cond_reduction",
                      "fdr_control_level", "small_sample_threshold", "con_sigma",
                      "high_contrib_fallback_percentile"]:
            print(f"{param}:")
            print(f"  Value: {explanations[param]['Value']}")
            print(f"  Explanation: {explanations[param]['Explanation']}\n")

        print("=" * 50 + "\n")


config = CausalConfig()
print_lock = Lock()


def bootstrap_worker(args):
    boot_indices, features, is_numeric, target_name, current_alpha, max_path_length = args
    boot_features = [f.copy_with_data(f.data[boot_indices]) for f in features]

    if is_numeric:
        boot_corr = np.corrcoef([f.data for f in boot_features])
    else:
        boot_corr, _ = spearmanr([f.data for f in boot_features])

    boot_paths, _ = _build_with_alpha(
        boot_features, boot_corr, target_name, current_alpha,
        is_numeric, skip_prune=True, max_path_length=max_path_length
    )
    return [tuple(p) for p in boot_paths]


def process_alpha(args):
    alpha, features, corr_matrix, target_name, is_numeric, max_path_length = args

    paths, dag = _build_with_alpha(
        features, corr_matrix, target_name, alpha, is_numeric,
        max_path_length=max_path_length
    )
    return alpha, paths, dag


class Feature:
    """封装节点（特征/目标变量）的核心属性"""

    def __init__(self, name: str, feature_names: list, all_con_values=None):
        self.name = name
        self.data = shared_globals.data[name].values

        try:
            self.data = np.asarray(self.data, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Feature {name} contains non-numeric data: {e}") from e
        mean = np.mean(self.data)
        std = np.std(self.data)

        if std < 1e-9:
            self.noise = 0.0
            with print_lock:
                print(f"Feature {name} is a constant feature, noise forced to 0")
        else:

            denom = np.abs(mean) + 1e-6
            self.noise = std / denom

            if self.noise > 1e6:
                self.noise = 1e6
                with print_lock:
                    print(f"Feature {name} has excessively large noise, truncated to 1e6")

        self.con_i = (shared_globals.con_list[feature_names.index(name)]
                      if name in feature_names else 0.0)
        self.all_con_values = all_con_values

        self.is_high_contrib = False
        self.threshold = None
        self.judgment_reason = ""

        if feature_names and name in feature_names and self.all_con_values is not None:

            if len(self.all_con_values) >= config.small_sample_threshold:
                self.is_high_contrib, self.threshold, self.judgment_reason = self._is_high_contrib_data_driven(
                    self.all_con_values)
            else:

                threshold = np.percentile(self.all_con_values, 80)
                self.threshold = threshold
                self.is_high_contrib = self.con_i >= threshold
                self.judgment_reason = f"Small sample size ({len(self.all_con_values)} < {config.small_sample_threshold}), using 80th percentile threshold: {threshold:.4f}"

    def _is_high_contrib_data_driven(self, all_con_values):
        """基于贡献值分布的密度拐点确定高贡献特征，返回(是否高贡献, 阈值, 原因)"""

        kde = gaussian_kde(all_con_values, bw_method='silverman')
        sorted_con = np.sort(all_con_values)
        density = kde(sorted_con)

        from scipy.ndimage import gaussian_filter1d
        density_smoothed = gaussian_filter1d(density, sigma=1)

        first_deriv = np.gradient(density_smoothed)
        second_deriv = np.gradient(first_deriv)
        sign_changes = np.diff(np.sign(second_deriv))
        inflection_points = np.where(sign_changes != 0)[0]

        if len(inflection_points) > 0:

            right_inflections = [i for i in inflection_points if i > len(sorted_con) * 0.3]
            if right_inflections:
                threshold_idx = right_inflections[-1]
                threshold = sorted_con[threshold_idx]
                max_con = max(all_con_values)
                adjusted_threshold = min(threshold, max_con * 0.9)

                reason = f"Using right inflection point threshold: {threshold:.4f}, adjusted to: {adjusted_threshold:.4f}"
                return self.con_i >= adjusted_threshold, adjusted_threshold, reason

        threshold = np.percentile(all_con_values, config.high_contrib_fallback_percentile)
        reason = f"No valid inflection points found, using {config.high_contrib_fallback_percentile}th percentile threshold: {threshold:.4f}"
        return self.con_i >= threshold, threshold, reason

    def copy_with_data(self, new_data):
        new_feature = Feature(self.name, [], None)
        new_feature.data = new_data
        new_feature.noise = self.noise
        new_feature.con_i = self.con_i
        new_feature.is_high_contrib = self.is_high_contrib
        new_feature.threshold = self.threshold
        new_feature.judgment_reason = self.judgment_reason
        return new_feature


def _find_paths(dag, target_idx, max_length=3):
    paths = []
    start_nodes = [i for i in range(dag.shape[0]) if i != target_idx and dag[i, target_idx] == 1]
    if not start_nodes and max_length == 1:
        return []

    from collections import deque
    queue = deque()
    for start in start_nodes:
        queue.append((start, [start], 1))

    while queue:
        current, path, length = queue.popleft()
        if current == target_idx:
            paths.append(path.copy())
            continue
        if length >= max_length:
            continue
        next_nodes = [j for j in range(dag.shape[0]) if j != current and dag[current, j] == 1 and j not in path]
        for next_node in next_nodes:
            new_path = path.copy()
            new_path.append(next_node)
            queue.append((next_node, new_path, length + 1))
    return paths


@lru_cache(maxsize=10)
def cached_make_skeleton(n, sample_num, alpha, corr_matrix_tuple, feature_con, feature_noise,
                         is_high_contrib, contrib_values, enable_progress=True):
    adj = np.ones((n, n)) - np.eye(n)
    sep_sets = [[[] for _ in range(n)] for _ in range(n)]
    l = 0
    corr_matrix = np.array(corr_matrix_tuple)

    @lru_cache(maxsize=None)
    def cached_is_independent(i, j, cond_tuple):
        cond = list(cond_tuple)
        idx = [i, j] + cond
        corr_sub = corr_matrix[np.ix_(idx, idx)]
        corr_sub = np.nan_to_num(corr_sub, nan=0.0, posinf=1.0, neginf=-1.0)

        eps = 1e-4
        corr_sub_reg = corr_sub + eps * np.eye(corr_sub.shape[0])

        try:

            partial_corr = np.linalg.pinv(corr_sub_reg, rcond=1e-3)
        except np.linalg.LinAlgError:

            partial_corr = np.eye(corr_sub_reg.shape[0]) * (1 / eps)

        denominator = np.sqrt(abs(partial_corr[0, 0] * partial_corr[1, 1]))
        rho = (-partial_corr[0, 1]) / denominator if denominator >= 1e-10 else 0.0
        rho = np.clip(rho, -0.99999, 0.99999)

        con_i, con_j = feature_con[i], feature_con[j]
        noise_i, noise_j = feature_noise[i], feature_noise[j]
        con_penalty = np.exp(-(con_i - con_j) ** 2 / (2 * config.con_sigma ** 2))
        noise_penalty = np.exp(-0.5 * (noise_i + noise_j))
        sample_penalty = min(1.0, sample_num / 1000)
        omega = con_penalty * noise_penalty * sample_penalty

        t_stat = (rho / np.sqrt((1 - rho ** 2) / (sample_num - len(cond) - 2))) * omega
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))

        if is_high_contrib[i] or is_high_contrib[j]:
            max_con = max(contrib_values) if contrib_values else 1.0
            rel_con_i = con_i / max_con if max_con > 0 else 0
            rel_con_j = con_j / max_con if max_con > 0 else 0
            adjust_factor = 1 - (max(rel_con_i, rel_con_j) * config.max_alpha_reduction)
            adjusted_alpha = alpha * adjust_factor

            return p_value >= adjusted_alpha

        return p_value >= alpha

    while True:
        updated = False
        pairs = combinations(range(n), 2)
        total_pairs = n * (n - 1) // 2

        for i, j in pairs:
            if adj[i, j] == 0:
                continue
            neighbors = [k for k in range(n) if adj[i, k] == 1 and k != j]
            if len(neighbors) >= l:

                if is_high_contrib[i] or is_high_contrib[j]:

                    max_con = max(contrib_values) if contrib_values else 1.0
                    rel_con = max(feature_con[i], feature_con[j]) / max_con if max_con > 0 else 0
                    cond_reduction = int(round(rel_con * config.max_cond_reduction))
                    max_cond = max(0, l - cond_reduction)




                else:
                    max_cond = l

                for cond in combinations(neighbors, max_cond):
                    if cached_is_independent(i, j, cond):
                        adj[i, j] = adj[j, i] = 0
                        sep_sets[i][j] = sep_sets[j][i] = list(cond)
                        updated = True
                        break
        if not updated and l < n:
            l += 1
        else:
            break
    return (adj.tolist(), [sublist for sublist in sep_sets])


def _build_with_alpha(features, corr_matrix, target_name, alpha, is_numeric, skip_prune=False, max_path_length=3):
    sample_num = len(features[0].data)
    n = len(features)

    feature_con = tuple(f.con_i for f in features)
    feature_noise = tuple(f.noise for f in features)
    is_high_contrib = tuple(f.is_high_contrib for f in features)
    contrib_values = tuple(f.con_i for f in features if f.con_i > 0)
    corr_matrix_tuple = tuple(map(tuple, corr_matrix))

    enable_progress = threading.current_thread() is threading.main_thread()

    skeleton_list, sep_sets = cached_make_skeleton(
        n, sample_num, alpha, corr_matrix_tuple, feature_con, feature_noise,
        is_high_contrib, contrib_values, enable_progress
    )
    skeleton = np.array(skeleton_list)

    dag = extend_to_dag(skeleton, sep_sets, n)

    if skip_prune:
        target_idx = shared_globals.all_names.index(target_name[0])
        valid_paths = _find_paths(dag, target_idx, max_length=max_path_length)
    else:
        valid_paths = prune_paths(dag, features, shared_globals.all_names, target_name,
                                  alpha, is_numeric, max_path_length=max_path_length)

    return valid_paths, dag


def build_causal_graph(features, corr_matrix, target_name, is_numeric,
                       alpha_list=None, max_path_length=None, ):
    n_bootstrap = config.n_bootstrap
    con_percentile = config.con_percentile
    noise_percentile = config.noise_percentile
    min_valid_paths = config.min_valid_paths

    config.auto_tune_based_on_data(features)
    max_path_length = max_path_length or config.max_path_length

    alpha_list = alpha_list or [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    best_alpha = max(alpha_list)

    with print_lock:
        print("=" * 60)
        print("           Causal Graph Construction Key Parameters Summary")
        print("=" * 60)
        print(f"Target Variable: {target_name}")
        print(f"Total Number of Features: {len(features)}")
        print(
            f"Used Alpha Threshold List: {alpha_list or [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]}")
        print(f"Maximum Path Length Limit: {max_path_length}")
        print(f"Number of Bootstrap Samples: {n_bootstrap}")
        print(f"Minimum Valid Path Count Threshold: {min_valid_paths}")
        print("-" * 60)

    shared_globals.target_idx = shared_globals.all_names.index(target_name[0])

    parallel_args = [
        (alpha, features, corr_matrix, target_name, is_numeric, max_path_length)
        for alpha in alpha_list
    ]

    with Pool(processes=min(os.cpu_count(), len(alpha_list))) as pool:
        results = list(tqdm(
            pool.imap(process_alpha, parallel_args),
            total=len(alpha_list), desc="Threshold Search Progress"
        ))

    best_paths, best_dag = [], None
    for alpha, paths, dag in results:
        current_count = len(paths)
        with print_lock:
            print(f"alpha={alpha:.3f} Number of Valid Paths: {current_count}")
        if current_count >= min_valid_paths and current_count > len(best_paths):
            best_paths, best_dag = paths, dag
            if current_count >= 10:
                with print_lock:
                    print("Sufficient paths found, terminating threshold search early")
                break

    if not best_paths:
        with print_lock:
            print("Insufficient paths found, retrying with maximum alpha")
        best_paths, best_dag = _build_with_alpha(
            features, corr_matrix, target_name, max(alpha_list), is_numeric,
            max_path_length=max_path_length
        )

    if not best_paths:
        best_alpha = max(alpha_list)
        best_paths, best_dag = _build_with_alpha(
            features, corr_matrix, target_name, best_alpha, is_numeric,
            max_path_length=max_path_length
        )

    high_contrib_thresholds = [f.threshold for f in features if f.is_high_contrib]
    overall_high_contrib_threshold = np.mean(high_contrib_thresholds) if high_contrib_thresholds else 0

    target_idx = shared_globals.target_idx
    path_feature_indices = set(idx for path in best_paths for idx in path if idx != target_idx)
    filtered_features = [shared_globals.all_names[i] for i in path_feature_indices]
    filtered_features.sort()
    shared_globals.filtered_features = filtered_features

    with print_lock:
        print(f"\n【Features Involved in Valid Paths】Total: {len(filtered_features)}:")
        for i, feat in enumerate(filtered_features, 1):
            print(f"{i}. {feat}")

    draw_causal_graph(best_paths, features, shared_globals.all_names, target_name)
    classify_paths(best_paths, shared_globals.all_names, target_name)
    shared_globals.valid_paths = best_paths

    with print_lock:
        print("\n" + "=" * 60)
        print("           Final Results of Causal Graph Construction")
        print("=" * 60)
        print(f"1. Optimal Alpha Threshold: {best_alpha:.4f}")
        print(f"2. Total Number of Valid Paths: {len(best_paths)}")
        print(f"   - Number of Direct Paths: {len([p for p in best_paths if len(p) - 1 == 1])}")
        print(f"   - Number of Indirect Paths: {len([p for p in best_paths if len(p) - 1 > 1])}")
        print(f"3. High Contribution Feature Determination Threshold (Average): {overall_high_contrib_threshold:.4f}")
        print(f"4. Number of High Contribution Features: {sum(1 for f in features if f.is_high_contrib)}")
        print(f"5. Path Stability FDR Control Level: {config.fdr_control_level}")
        print(f"6. Minimum Stability Threshold: {config.min_stability_threshold}")
        print("=" * 60 + "\n")

    high_contrib_features = get_and_print_high_contrib_features(features, print_lock)
    shared_globals.high_contrib_features = high_contrib_features

    return best_paths, best_dag, shared_globals.all_names


def _compute_independence(i, j, cond, corr_matrix, sample_num,
                          feature_con, feature_noise, is_high_contrib, alpha,
                          contrib_values, con_sigma=0.5):
    idx = [i, j] + cond
    corr_sub = corr_matrix[np.ix_(idx, idx)]
    corr_sub = np.nan_to_num(corr_sub, nan=0.0, posinf=1.0, neginf=-1.0)

    eps = 1e-4
    corr_sub_reg = corr_sub + eps * np.eye(corr_sub.shape[0])
    try:
        partial_corr = np.linalg.pinv(corr_sub_reg, rcond=1e-3)
    except np.linalg.LinAlgError:
        partial_corr = np.eye(corr_sub_reg.shape[0]) * (1 / eps)

    denominator = np.sqrt(abs(partial_corr[0, 0] * partial_corr[1, 1]))
    rho = (-partial_corr[0, 1] / denominator) if denominator >= 1e-10 else 0.0
    rho = np.clip(rho, -0.99999, 0.99999)

    con_i, con_j = feature_con[i], feature_con[j]
    noise_i, noise_j = feature_noise[i], feature_noise[j]
    con_penalty = np.exp(-(con_i - con_j) ** 2 / (2 * con_sigma ** 2))
    noise_penalty = np.exp(-0.5 * (noise_i + noise_j))
    sample_penalty = min(1.0, sample_num / 1000)
    omega = con_penalty * noise_penalty * sample_penalty

    t_stat = (rho / np.sqrt((1 - rho ** 2) / (sample_num - len(cond) - 2))) * omega
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    if is_high_contrib[i] or is_high_contrib[j]:
        max_con = max(contrib_values) if contrib_values else 1.0
        rel_con_i = con_i / max_con if max_con > 0 else 0
        rel_con_j = con_j / max_con if max_con > 0 else 0
        adjust_factor = 1 - (max(rel_con_i, rel_con_j) * config.max_alpha_reduction)
        adjusted_alpha = alpha * adjust_factor
        return p_value >= adjusted_alpha
    return p_value >= alpha


def is_independent(i, j, cond, features, corr_matrix, sample_num, alpha):
    feature_con = tuple(f.con_i for f in features)
    feature_noise = tuple(f.noise for f in features)
    is_high_contrib = tuple(f.is_high_contrib for f in features)
    contrib_values = tuple(f.con_i for f in features if f.con_i > 0)
    return _compute_independence(
        i, j, cond, corr_matrix, sample_num,
        feature_con, feature_noise, is_high_contrib, alpha, contrib_values
    )


def extend_to_dag(skeleton, sep_sets, n):
    dag = skeleton.copy()

    for i, j, k in combinations(range(n), 3):
        if dag[i, j] == 1 and dag[j, k] == 1 and dag[i, k] == 0:
            if j not in sep_sets[i][k]:
                dag[j, i] = dag[j, k] = 0

    for i in range(n):
        for j in range(n):
            if i != j and dag[i, j] == 1 and dag[j, i] == 0:
                for k in range(n):
                    if k != i and k != j and dag[j, k] == 1 and dag[k, j] == 1:
                        if dag[i, k] == 0 and dag[k, i] == 0:
                            dag[k, j] = 0
    return dag


from concurrent.futures import ThreadPoolExecutor, as_completed


def prune_paths(dag, features, all_names, target_name, current_alpha, is_numeric,
                max_path_length=3, n_bootstrap=30, con_percentile=30, noise_percentile=70):
    target_idx = all_names.index(target_name[0])
    paths = _find_paths(dag, target_idx, max_length=max_path_length)
    original_paths = [tuple(path) for path in paths]

    if not original_paths:
        return []

    high_contrib_indices = [i for i, f in enumerate(features) if f.is_high_contrib]

    sample_size = len(features[0].data)
    boot_indices_list = [np.random.choice(sample_size, size=sample_size, replace=True)
                         for _ in range(n_bootstrap)]

    args_list = [
        (boot_indices, features, is_numeric, target_name, current_alpha, max_path_length)
        for boot_indices in boot_indices_list
    ]

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), n_bootstrap)) as executor:
        with print_lock:
            progress_bar = tqdm(total=n_bootstrap, desc="Bootstrap Sampling Validation")
        boot_results = []
        futures = [executor.submit(bootstrap_worker, args) for args in args_list]
        for future in as_completed(futures):
            boot_results.append(future.result())
            with print_lock:
                progress_bar.update(1)
        with print_lock:
            progress_bar.close()

    path_freq = defaultdict(int)
    for res in boot_results:
        for p in res:
            if p in original_paths:
                path_freq[p] += 1

    stable_paths = []
    if original_paths:

        frequencies = np.array([path_freq.get(p, 0) / n_bootstrap for p in original_paths])
        path_indices = np.arange(len(original_paths))

        sorted_indices = np.argsort(frequencies)[::-1]
        sorted_freq = frequencies[sorted_indices]

        fdr_values = (sorted_freq * len(frequencies)) / (np.arange(len(frequencies)) + 1)

        significant_mask = fdr_values <= config.fdr_control_level

        if np.any(significant_mask):

            min_significant_freq = sorted_freq[significant_mask][-1]

            min_significant_freq = max(min_significant_freq, config.min_stability_threshold)

            for i, path in enumerate(original_paths):
                freq = path_freq.get(path, 0) / n_bootstrap
                path_names = [all_names[i] for i in path]
                has_high_contrib = any(idx in high_contrib_indices for idx in path
                                       if idx != target_idx)

                adjusted_threshold = max(min_significant_freq * 0.8, config.min_stability_threshold)

                if freq >= min_significant_freq or (has_high_contrib and freq >= adjusted_threshold):
                    stable_paths.append(list(path))







        else:

            for path in original_paths:
                freq = path_freq.get(path, 0) / n_bootstrap
                path_names = [all_names[i] for i in path]
                if freq >= config.min_stability_threshold:
                    stable_paths.append(list(path))

    all_con = [f.con_i for f in features if f.name != target_name[0]]
    all_noise = [f.noise for f in features if f.name != target_name[0]]
    if not all_con or not all_noise:
        return []

    con_threshold = np.percentile(all_con, con_percentile)
    noise_threshold = np.percentile(all_noise, noise_percentile)

    valid = []
    for path in stable_paths:
        if path[-1] != target_idx or not (1 <= len(path) - 1 <= max_path_length):
            continue

        feature_indices = [idx for idx in path if idx != target_idx]
        if not feature_indices:
            continue

        mean_con = np.mean([features[idx].con_i for idx in feature_indices])
        has_high_contrib = any(idx in high_contrib_indices for idx in feature_indices)
        path_names = [all_names[i] for i in path]

        if has_high_contrib:
            con_distribution = np.array(all_con)
            con_iqr = np.percentile(con_distribution, 75) - np.percentile(con_distribution, 25)

            adjustment = 0.5 + (con_iqr / (np.mean(con_distribution) + 1e-8)) * 0.2
            adjusted_con_threshold = con_threshold * min(0.8, adjustment)

        middle_noise = np.mean([features[i].noise for i in path[1:-1]]) if len(path) >= 3 else 0
        if has_high_contrib:
            noise_distribution = np.array(all_noise)
            noise_iqr = np.percentile(noise_distribution, 75) - np.percentile(noise_distribution, 25)
            adjustment = 1.2 + (noise_iqr / (np.mean(noise_distribution) + 1e-8)) * 0.3
            adjusted_noise_threshold = noise_threshold * min(1.5, adjustment)

            if middle_noise <= adjusted_noise_threshold:
                valid.append(path)





        else:

            if middle_noise <= noise_threshold:
                valid.append(path)

    return valid


def draw_causal_graph(valid_paths, features, all_names, target_name):
    G = nx.DiGraph()
    target_node = target_name[0]
    G.add_nodes_from(all_names)

    for path in valid_paths:
        for i in range(len(path) - 1):
            G.add_edge(all_names[path[i]], all_names[path[i + 1]])

    pos = {target_node: (0, 0)}
    others = [n for n in all_names if n != target_node]
    angle = 2 * np.pi / len(others) if others else 0
    pos.update({others[i]: (2 * np.cos(i * angle), 2 * np.sin(i * angle))
                for i in range(len(others))})

    all_con = [f.con_i for f in features]
    max_con = max(all_con) if all_con and max(all_con) != 0 else 1.0
    node_size = [f.con_i / max_con * 2000 for f in features]
    node_colors = ['lightgreen' if f.is_high_contrib else 'lightblue' for f in features]

    edges = G.edges()
    edge_colors = [(features[all_names.index(u)].noise + features[all_names.index(v)].noise) / 2
                   for u, v in edges]
    edge_colors = [plt.cm.RdYlGn(1 - min(n, 0.6) / 0.6) for n in edge_colors]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                           arrowstyle='->', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos,
                            labels={n: f"{n}\n{features[i].con_i:.5f}"
                                    for i, n in enumerate(all_names)},
                            font_size=9)

    plt.title(
        "Causal Feature Graph (Green Nodes = High Contribution Features, Node Size = Contribution Value, Edge Color = Noise)")

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def classify_paths(valid_paths, all_names, target_name):
    direct, indirect = [], []
    for path in valid_paths:
        path_str = " → ".join([all_names[i] for i in path])
        if len(path) - 1 == 1:
            direct.append(path_str)
        else:
            indirect.append(path_str)

    shared_globals.direct_paths = direct
    shared_globals.indirect_paths = indirect

    with print_lock:
        print("\n【Direct Paths (Feature→Target)】")
        for i, p in enumerate(direct, 1):
            print(f"{i}. {p}")
        print("\n【Indirect Paths (Feature→...→Target)】")
        for i, p in enumerate(indirect, 1):
            print(f"{i}. {p}")


if __name__ == '__main__':
    if not hasattr(shared_globals, 'feature_max_abs'):
        raise ValueError("Please run the feature perturbation module first")

    all_con_values = [shared_globals.con_list[i] for i in range(len(shared_globals.feature_names))]
    with print_lock:
        print(
            f"\nList of All Feature Contribution Values: {[(shared_globals.feature_names[i], round(all_con_values[i], 4)) for i in range(len(all_con_values))]}")
        print(
            f"Contribution Value Sample Size: {len(all_con_values)}, Small Sample Threshold: {config.small_sample_threshold}")

    features = [Feature(name=n, feature_names=shared_globals.feature_names, all_con_values=all_con_values)
                for n in shared_globals.all_names]

    is_numeric = all(f.data.dtype.kind in 'iufc' for f in features)
    with print_lock:
        print(f"\nData Type Check: {'Numeric' if is_numeric else 'Non-numeric'}")

    if is_numeric:
        corr_matrix = np.corrcoef([f.data for f in features])
    else:
        corr_matrix, _ = spearmanr([f.data for f in features])

    valid_paths, dag, _ = build_causal_graph(features, corr_matrix,
                                             shared_globals.target_name,
                                             is_numeric)