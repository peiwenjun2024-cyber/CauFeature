import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import json
from pathlib import Path

import shared_globals

import time
from typing import List, Tuple
from tensorflow.keras.models import load_model
from scipy.stats import entropy, skew


MAX_PERTURBATION_POINTS = 101
SEMANTIC_METADATA_FILENAME = "feature_semantic_metadata.json"
BASELINE_BY_SEMANTIC_TYPE = {
    "categorical": "mode",
    "ordinal": "mode",
    "count": "mean",
    "measurement": "mean",
}
SUPPORTED_PERTURBATION_POLICIES = {
    "observed_model_support",
    "observed_model_support_capped",
    "continuous_profile_quantile_or_linspace",
}


def load_semantic_metadata(dataset_name, feature_names, metadata_path=None):
    """Load the authoritative baseline + perturbation metadata for one dataset."""
    if not dataset_name:
        raise RuntimeError("dataset_name is required for semantic metadata")
    path = Path(metadata_path) if metadata_path else Path(__file__).resolve().with_name(SEMANTIC_METADATA_FILENAME)
    if not path.exists():
        raise FileNotFoundError(f"Semantic metadata not found: {path}")
    obj = json.loads(path.read_text())
    rows = [r for r in obj.get("records", []) if r.get("dataset") == dataset_name]
    by_feature = {r["feature"]: r for r in rows}
    expected = list(feature_names)
    missing = [f for f in expected if f not in by_feature]
    extra = [f for f in by_feature if f not in expected]
    if missing or extra or len(rows) != len(by_feature):
        raise RuntimeError(
            f"Semantic metadata mismatch for {dataset_name}: "
            f"missing={missing}, extra={extra}, duplicate_rows={len(rows)-len(by_feature)}"
        )

    semantic_types = {}
    baseline_policy = {}
    perturbation_classes = {}
    perturbation_policy = {}
    for feature in expected:
        rec = by_feature[feature]
        semantic = rec.get("semantic_type")
        if semantic not in BASELINE_BY_SEMANTIC_TYPE:
            raise RuntimeError(
                f"Unsupported semantic_type={semantic!r} for {dataset_name}:{feature}"
            )
        semantic_types[feature] = semantic

        expected_baseline = BASELINE_BY_SEMANTIC_TYPE[semantic]
        configured_baseline = rec.get("baseline_policy_recommended")
        if configured_baseline != expected_baseline:
            raise RuntimeError(
                f"Metadata baseline-policy mismatch for {dataset_name}:{feature}: "
                f"semantic_type={semantic}, expected={expected_baseline}, "
                f"configured={configured_baseline}"
            )
        baseline_policy[feature] = expected_baseline

        pclass = rec.get("perturbation_semantic_class")
        ppolicy = rec.get("perturbation_policy_recommended")
        if not pclass:
            raise RuntimeError(
                f"Missing perturbation_semantic_class for {dataset_name}:{feature}"
            )
        if ppolicy not in SUPPORTED_PERTURBATION_POLICIES:
            raise RuntimeError(
                f"Unsupported perturbation policy={ppolicy!r} for {dataset_name}:{feature}"
            )
        perturbation_classes[feature] = pclass
        perturbation_policy[feature] = ppolicy

    return (
        semantic_types,
        baseline_policy,
        perturbation_classes,
        perturbation_policy,
        path,
    )


def _outlier_ratio(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float(np.mean((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)))


def feature_profile(values):
    """Descriptive numeric profile only; never infers semantic type."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    unique = np.unique(x)
    n = len(x)
    k = len(unique)
    value_range = float(unique[-1] - unique[0]) if k else 0.0
    return {
        "n": n,
        "nunique": k,
        "unique_ratio": k / max(n, 1),
        "range": value_range,
        "skewness": float(skew(x)) if n > 2 and np.std(x) > 0 else 0.0,
        "outlier_ratio": _outlier_ratio(x),
    }


def _observed_support_candidates(values, max_points=MAX_PERTURBATION_POINTS):
    """Return only values actually observed by the model, capped by empirical quantiles."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.asarray([], dtype=float), "empty"
    unique = np.unique(x)
    if len(unique) <= max_points:
        return unique.astype(float), "observed_model_support"

    # P1 takes a maximum effect, not an expectation. Cover the whole legal
    # support rather than weighting candidates by empirical frequency. Select
    # evenly by rank over sorted unique observed values and always include
    # both support extremes; no synthetic numeric state is created.
    idx = np.rint(np.linspace(0, len(unique) - 1, max_points)).astype(int)
    result = unique[idx].astype(float)
    return result, "observed_model_unique_support_rank_capped"


def _continuous_profile_candidates(values, max_points=MAX_PERTURBATION_POINTS):
    """Bounded candidates for a source-defined continuous measurement."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    profile = feature_profile(x)
    if len(x) == 0:
        return np.asarray([], dtype=float), "empty", profile
    if abs(profile["skewness"]) > 2.0 or profile["outlier_ratio"] > 0.02:
        q = np.linspace(0.0, 100.0, max_points)
        result = np.unique(np.percentile(x, q)).astype(float)
        strategy = "bounded_quantile_continuous"
    else:
        result = np.unique(
            np.linspace(float(np.min(x)), float(np.max(x)), max_points)
        ).astype(float)
        strategy = "bounded_linspace_continuous"
    return result, strategy, profile


class CausalFeaturePerturbation:
    """Feature perturbation module adapted to CausalFeatureX framework (parallel computing optimized version)"""

    def __init__(
        self,
        model: tf.keras.Model,
        data: pd.DataFrame,
        dataset_name: str = None,
        semantic_metadata_path=None,
    ):
        self.model = model
        self.data = data
        self.sample_size = data.shape[0]
        self.dataset_name = dataset_name
        (
            self.semantic_types,
            self.baseline_policy,
            self.perturbation_classes,
            self.perturbation_policy,
            self.semantic_metadata_path,
        ) = load_semantic_metadata(
            dataset_name,
            shared_globals.feature_names,
            metadata_path=semantic_metadata_path,
        )
        self.baseline = self._calculate_baseline()
        self.baseline_np = self.baseline.values.astype(np.float32).copy()
        self.n_features = len(shared_globals.feature_names)

        self.baseline_reshaped = self.baseline_np.reshape(-1, self.n_features, 1)

        self.baseline_pred = self.model.predict(
            self.baseline_reshaped,
            batch_size=128,
            verbose=0
        )

        self.baseline_pred_dist = self.baseline_pred[0]

        self.feature_values = {}
        for col in shared_globals.feature_names:
            try:
                self.feature_values[col] = sorted(pd.to_numeric(data[col], errors='coerce').dropna().unique().tolist())
            except:
                self.feature_values[col] = []

        shared_globals.baselines = self.baseline
        shared_globals.baseline_preds = self.baseline_pred

    def reference_baseline_vector(self, row_indices=None):
        """Build a semantically valid reference row.

        semantic categorical/ordinal -> empirical mode (an observed state)
        semantic count/measurement     -> empirical mean

        The semantic class is loaded from feature_semantic_metadata.json and does
        not depend on dtype, nunique, integer-like checks, or the current seed.
        row_indices is used by P1 bootstrap reliability so bootstrap baselines
        obey exactly the same reference policy as the original baseline.
        """
        encoded_frame = shared_globals.feature_data
        raw_frame = getattr(shared_globals, "raw_feature_data", None)
        if row_indices is not None:
            encoded_frame = encoded_frame.iloc[row_indices]
            if raw_frame is not None:
                raw_frame = raw_frame.iloc[row_indices]

        values = []
        for feature in shared_globals.feature_names:
            policy = self.baseline_policy[feature]
            encoded = pd.to_numeric(
                encoded_frame[feature], errors="coerce"
            ).dropna()

            if policy == "mode":
                # Use model-space encoded values so textual categories map exactly
                # to the values seen by the selector CNN.
                mode = encoded.mode(dropna=True)
                value = float(mode.iloc[0]) if not mode.empty else np.nan
            elif policy == "mean":
                value = float(encoded.mean()) if not encoded.empty else np.nan
            else:
                raise RuntimeError(
                    f"Unsupported baseline policy {policy!r} for {feature}"
                )

            if not np.isfinite(value):
                # Only a defensive fallback; current production datasets have
                # no all-missing feature columns.
                global_encoded = pd.to_numeric(
                    shared_globals.feature_data[feature], errors="coerce"
                ).dropna()
                value = float(global_encoded.mean()) if not global_encoded.empty else 0.0
            values.append(np.float32(value))
        return np.asarray(values, dtype=np.float32)

    def _calculate_baseline(self):
        vector = self.reference_baseline_vector()
        return pd.DataFrame([vector], columns=shared_globals.feature_names)

    def get_col_ranges(self, col_name: str) -> Tuple[float, float, int, List[float]]:
        """Optimized feature statistics calculation"""
        s = shared_globals.data[col_name]
        numeric_series = pd.to_numeric(s, errors='coerce').dropna()

        if numeric_series.empty:
            return (0.0, 0.0, 0, [])

        min_val = numeric_series.min()
        max_val = numeric_series.max()
        unique_vals = numeric_series.unique()

        return (
            min_val,
            max_val,
            len(unique_vals),
            sorted(unique_vals.tolist())
        )

    def value_range(self):
        """Parallel calculation of statistical ranges for all features"""
        features = list(shared_globals.feature_data.columns)
        results = [None] * len(features)

        def process_feature(col):
            return self.get_col_ranges(col)

        max_workers = os.cpu_count() * 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_feature, col): i for i, col in enumerate(features)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Feature {features[idx]} processing failed: {e}")
                    results[idx] = (0.0, 0.0, 0, [])

        min_list = [res[0] for res in results]
        max_list = [res[1] for res in results]
        count_list = [res[2] for res in results]
        range_list = [res[3] for res in results]
        return min_list, max_list, count_list, range_list

    def disturbance_range(self, min_list=None, max_list=None, count_list=None, range_list=None):
        """Generate perturbations only from source-backed perturbation metadata."""
        x_list = []
        raw_frame = getattr(shared_globals, "raw_feature_data", None)

        for col in tqdm(
            shared_globals.feature_names,
            total=len(shared_globals.feature_names),
            desc="Generating metadata-defined perturbation values",
        ):
            encoded = pd.to_numeric(
                shared_globals.feature_data[col], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            raw_series = (
                raw_frame[col]
                if raw_frame is not None and col in raw_frame.columns
                else shared_globals.feature_data[col]
            )
            pclass = self.perturbation_classes[col]
            policy = self.perturbation_policy[col]

            if policy == "observed_model_support":
                candidates = np.unique(encoded).astype(float)
                profile = feature_profile(encoded)
                strategy = "observed_model_support"
            elif policy == "observed_model_support_capped":
                candidates, strategy = _observed_support_candidates(
                    encoded, MAX_PERTURBATION_POINTS
                )
                profile = feature_profile(encoded)
            elif policy == "continuous_profile_quantile_or_linspace":
                raw_numeric = pd.to_numeric(
                    raw_series, errors="coerce"
                ).dropna().to_numpy(dtype=float)
                candidates, strategy, profile = _continuous_profile_candidates(
                    raw_numeric, MAX_PERTURBATION_POINTS
                )
            else:
                raise RuntimeError(
                    f"Unsupported perturbation policy {policy!r} for {col}"
                )

            print(
                f"Feature {col}: perturb_class={pclass}, "
                f"baseline={self.baseline_policy[col]}, policy={policy}, "
                f"strategy={strategy}, n={profile['n']}, "
                f"nunique={profile['nunique']}, "
                f"unique_ratio={profile['unique_ratio']:.6g}, "
                f"skew={profile.get('skewness', 0.0):.4g}, "
                f"outlier_ratio={profile.get('outlier_ratio', 0.0):.4g}, "
                f"candidates={len(candidates)}"
            )
            x_list.append(np.asarray(candidates, dtype=float).tolist())
        return x_list

    def _js_divergence(self, p, q):
        """Calculate JS divergence between two probability distributions"""

        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        m = 0.5 * (p + q)

        return 0.5 * (entropy(p, m) + entropy(q, m))

    def _process_single_feature(self, j, col, values):
        """Process contribution calculation for a single feature using JS divergence"""
        try:

            valid_values = []
            for val in values:
                try:
                    valid_values.append(float(val))
                except (ValueError, TypeError):
                    continue

            if not valid_values:
                return j, [], 0.0

            batch_input = np.repeat(self.baseline_np, len(valid_values), axis=0)
            batch_input[:, j] = valid_values
            batch_input = batch_input.reshape(-1, self.n_features, 1).astype(np.float32)
            batch_tensor = tf.convert_to_tensor(batch_input)

            preds = self.model.predict(
                batch_tensor,
                batch_size=128,
                verbose=0
            )

            valid_diffs = []
            for dist in preds:
                js_div = self._js_divergence(dist, self.baseline_pred_dist)
                valid_diffs.append(js_div)

            con = max(valid_diffs) if valid_diffs else 0.0
            return j, valid_diffs, con

        except Exception as e:
            print(f"Error processing feature {col}: {str(e)}")
            return j, [], 0.0

    def calculate_contributions(self, x_list: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        """Calculate all single-feature perturbations through shared streaming GPU batches."""
        y_list = [[] for _ in shared_globals.feature_names]
        con_list = [0.0 for _ in shared_globals.feature_names]
        chunk_size = 4096
        jobs = []
        start_time = time.perf_counter()

        def flush(batch_jobs):
            if not batch_jobs:
                return
            batch = np.repeat(self.baseline_np, len(batch_jobs), axis=0)
            for row, (j, value) in enumerate(batch_jobs):
                batch[row, j] = value
            batch = batch.reshape(-1, self.n_features, 1).astype(np.float32)
            preds = self.model(batch, training=False)
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            for (j, _), dist in zip(batch_jobs, np.asarray(preds)):
                score = float(self._js_divergence(dist, self.baseline_pred_dist))
                y_list[j].append(score)
                if score > con_list[j]:
                    con_list[j] = score

        for j, values in enumerate(x_list):
            for value in values:
                try:
                    jobs.append((j, float(value)))
                except (TypeError, ValueError):
                    continue
                if len(jobs) >= chunk_size:
                    flush(jobs)
                    jobs.clear()
        flush(jobs)

        print(
            f"Total time for batched feature perturbation experiments: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )
        return y_list, con_list

    def calculate_joint_contributions(self, target_features, max_combinations=10000):
        """Calculate maximum JS divergence for multi-feature joint interventions"""
        if not target_features:
            return 0.0

        feat_indices = [shared_globals.feature_names.index(f) for f in target_features]
        feat_perturb = [self.perturb_values[f] for f in target_features]

        total_combinations = np.prod([len(p) for p in feat_perturb])
        if total_combinations > max_combinations:
            print(f"Too many combinations ({total_combinations}), sampling {max_combinations} combinations")
            sampled_combinations = []
            for _ in range(max_combinations):
                combo = [np.random.choice(p) for p in feat_perturb]
                sampled_combinations.append(combo)
        else:
            from itertools import product
            sampled_combinations = list(product(*feat_perturb))

        batch_input = np.repeat(self.baseline_np, len(sampled_combinations), axis=0)
        for i, idx in enumerate(feat_indices):
            values = [combo[i] for combo in sampled_combinations]
            batch_input[:, idx] = values

        batch_input_reshaped = batch_input.reshape(-1, self.n_features, 1).astype(np.float32)
        preds = self.model.predict(
            batch_input_reshaped,
            batch_size=128,
            verbose=0
        )

        max_js = 0.0
        for dist in preds:
            js_div = self._js_divergence(dist, self.baseline_pred_dist)
            if js_div > max_js:
                max_js = js_div
        return max_js

    def joint_effect_exceeds(self, target_features, required_threshold, max_combinations=10000, chunk_size=4096):
        """Threshold-aware pair evaluation with exact early exit.

        For <= max_combinations the Cartesian support is unchanged. For larger supports,
        the same legacy random sampling rule is materialized first, then evaluated in
        streaming batches. Returning early cannot change the yes/no recovery decision.
        """
        if not target_features:
            return False, 0.0
        feat_indices = [shared_globals.feature_names.index(f) for f in target_features]
        feat_perturb = [self.perturb_values[f] for f in target_features]
        if any(len(v) == 0 for v in feat_perturb):
            return False, 0.0
        total = int(np.prod([len(v) for v in feat_perturb]))
        if total > max_combinations:
            combinations_list = [
                tuple(np.random.choice(values) for values in feat_perturb)
                for _ in range(max_combinations)
            ]
        else:
            from itertools import product
            combinations_list = list(product(*feat_perturb))

        max_js = 0.0
        for start in range(0, len(combinations_list), chunk_size):
            chunk = combinations_list[start:start + chunk_size]
            batch = np.repeat(self.baseline_np, len(chunk), axis=0)
            for fi, feature_idx in enumerate(feat_indices):
                batch[:, feature_idx] = [combo[fi] for combo in chunk]
            batch = batch.reshape(-1, self.n_features, 1).astype(np.float32)
            preds = self.model(batch, training=False)
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            for dist in np.asarray(preds):
                score = float(self._js_divergence(dist, self.baseline_pred_dist))
                max_js = max(max_js, score)
                if score > required_threshold:
                    return True, max_js
        return False, max_js

    def visualization(self, x_list: List[List[float]], y_list: List[List[float]],
                      con_list: List[float]) -> List[float]:
        """Visualize feature perturbation effects (JS divergence is non-negative, no need for absolute value)"""
        max_abs_values = [0.0] * len(shared_globals.feature_names)

        top_indices = np.argsort(con_list)[-20:][::-1]
        for i in top_indices:
            col = shared_globals.feature_names[i]
            if not y_list[i]:
                continue

            max_abs = max(y_list[i])
            max_abs_values[i] = max_abs

        for i in range(len(shared_globals.feature_names)):
            if i not in top_indices and y_list[i]:
                max_abs_values[i] = max(y_list[i])

        return max_abs_values

    # def feature_important(self, con_list: List[float]):
    #     """Feature importance analysis (using JS divergence results)"""
    #     total_con = sum(con_list)
    #     if total_con == 0:
    #         return
    #
    #     con_list = [max(0.0, con) for con in con_list]
    #     total_con = sum(con_list)
    #
    #     top_indices = np.argsort(con_list)[-5:][::-1]
    #     merged_con = [con_list[i] for i in top_indices]
    #     merged_names = [shared_globals.feature_names[i] for i in top_indices]
    #     other_con = total_con - sum(merged_con)
    #     other_con = max(0.0, other_con)
    #
    #     merged_con.append(other_con)
    #     merged_names.append("others")
    #
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #     axs[0].bar(merged_names, merged_con, alpha=0.6)
    #     axs[0].tick_params(axis='x', rotation=70)
    #     axs[1].pie(merged_con, labels=merged_names, autopct='%1.1f%%')
    #     plt.tight_layout()
    #     plt.show()

    # def feature_important(self, con_list: List[float]):
    #     """Feature importance analysis (only bar chart, IEEE compliant)"""
    #     total_con = sum(con_list)
    #     if total_con == 0:
    #         return
    #
    #     con_list = [max(0.0, con) for con in con_list]
    #     total_con = sum(con_list)
    #
    #     # 取前5个重要特征
    #     top_indices = np.argsort(con_list)[-5:][::-1]
    #     merged_con = [con_list[i] for i in top_indices]
    #     merged_names = [shared_globals.feature_names[i] for i in top_indices]
    #     other_con = total_con - sum(merged_con)
    #     other_con = max(0.0, other_con)
    #
    #     merged_con.append(other_con)
    #     merged_names.append("others")
    #
    #     # IEEE 列宽尺寸：3.5英寸（88mm），高度适配
    #     fig, ax = plt.subplots(figsize=(3.5, 2.8))
    #
    #     # 绘制柱状图（线条艺术：宽度≥1pt，矢量渲染）
    #     bars = ax.bar(merged_names, merged_con, alpha=0.6, linewidth=1.2)
    #
    #     from matplotlib import font_manager
    #     # 强制所有文本使用8号 Times New Roman
    #     shared_globals.tnr_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    #     shared_globals.tnr_font = font_manager.FontProperties(fname=shared_globals.tnr_font_path, size=8)
    #     ax.tick_params(axis='x', rotation=70, labelsize=8)
    #     ax.set_xticklabels(merged_names, fontproperties=shared_globals.tnr_font)
    #     ax.set_yticklabels(ax.get_yticks(), fontproperties=shared_globals.tnr_font)
    #
    #     # 紧凑布局，适配小尺寸
    #     plt.tight_layout(pad=0.2)
    #
    #     # 保存为600dpi矢量图（PDF）+ 位图兜底，满足IEEE分辨率要求
    #     plt.savefig('feature_importance_bar_ieee.pdf', dpi=600, bbox_inches='tight', format='pdf')
    #     plt.savefig('feature_importance_bar_ieee.png', dpi=600, bbox_inches='tight', format='png')
    #     plt.show()

    def feature_important(self, con_list: List[float]):
        import matplotlib.ticker as mticker
        total_con = sum(con_list)
        if total_con == 0:
            return

        con_list = [max(0.0, con) for con in con_list]
        total_con = sum(con_list)

        # 取前5个重要特征
        top_indices = np.argsort(con_list)[-5:][::-1]
        merged_con = [con_list[i] for i in top_indices]
        merged_names = [shared_globals.feature_names[i] for i in top_indices]
        other_con = total_con - sum(merged_con)
        merged_con.append(other_con)
        merged_names.append("others")

        # IEEE列宽尺寸（3.5英寸），调高高度避免标签超出
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.bar(merged_names, merged_con, alpha=0.6, linewidth=1.2)

        # -------------------------- 坐标轴核心设置（兼容Matplotlib 3.8） --------------------------
        # 1. 横轴：Feature + 45度旋转（移除labelproperties，改用遍历标签设字体）
        ax.set_xlabel('Feature', fontproperties=shared_globals.tnr_font, fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=8)  # 仅控制旋转和字号，移除不兼容参数
        # 低版本兼容：逐个设置横轴刻度标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(shared_globals.tnr_font)

        # 2. 纵轴：Importance + 关闭科学计数法 + 格式化小数
        ax.set_ylabel('Importance', fontproperties=shared_globals.tnr_font, fontsize=8)
        ax.tick_params(axis='y', labelsize=8)  # 仅控制字号，移除不兼容参数
        # 低版本兼容：逐个设置纵轴刻度标签字体
        for label in ax.get_yticklabels():
            label.set_fontproperties(shared_globals.tnr_font)

        # 关闭科学计数法 + 控制刻度数量/格式
        ax.ticklabel_format(style='plain', axis='y')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))  # 控制刻度数量
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))  # 保留2位小数

        # 3. 调整Y轴范围，避免刻度空荡
        ax.set_ylim(0, max(merged_con) * 1.2)

        # -------------------------- 布局优化（避免标签超出） --------------------------
        plt.tight_layout(pad=0.8)  # 增大内边距，防止标签超出画布
        # 保存600dpi矢量图（满足IEEE分辨率要求）
        plt.savefig('feature_importance_bar_ieee.pdf', dpi=600, bbox_inches='tight')
        plt.show()


    def run_perturbation(self):
        """Integrate all steps"""
        min_list, max_list, count_list, range_list = self.value_range()
        x_list = self.disturbance_range(min_list, max_list, count_list, range_list)
        self.perturb_values = {
            shared_globals.feature_names[i]: x_list[i]
            for i in range(len(shared_globals.feature_names))
        }

        y_list, con_list = self.calculate_contributions(x_list)

        max_abs_values = self.visualization(x_list, y_list, con_list)
        self.feature_important(con_list)

        self.feature_max_abs = [(shared_globals.feature_names[i], max_abs_values[i])
                                for i in range(len(shared_globals.feature_names))]
        self.feature_max_abs.sort(key=lambda x: x[1], reverse=True)
        self.con_list = con_list

        shared_globals.con_list = self.con_list
        shared_globals.feature_max_abs = self.feature_max_abs
        shared_globals.x_list = x_list
        shared_globals.feature_values = self.feature_values
        shared_globals.perturbation_instance = self

    def initialize_original_preds(self, data: pd.DataFrame, model) -> None:
        """
        Calculate predicted values of original samples, baseline distribution, and JS divergence,
        then store them in global variables

        Parameters:
            data: Original dataset (contains feature columns and target column)
            model: Trained model (used for prediction)
        """

        feature_data = shared_globals.feature_data
        print(f"Feature data shape: {feature_data.shape}")

        features_np = feature_data.values
        features_reshaped = features_np.reshape(-1, features_np.shape[1], 1)

        try:
            original_preds = model.predict(features_reshaped, verbose=0)

            if original_preds.ndim == 1:
                original_preds = original_preds.reshape(-1, 1)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate original sample predictions: {e}") from e

        baseline_dist = np.mean(original_preds, axis=0)

        baseline_dist = baseline_dist / np.sum(baseline_dist)

        sample_js_divergences = []
        for pred_dist in original_preds:
            js = self._js_divergence(pred_dist, baseline_dist)
            sample_js_divergences.append(js)
        sample_js_divergences = np.array(sample_js_divergences)

        shared_globals.original_preds = original_preds
        shared_globals.baseline_dist = baseline_dist
        shared_globals.sample_js = sample_js_divergences
        shared_globals.mean_js = np.mean(sample_js_divergences)

        print(f"Original sample predictions stored in global variables, shape: {original_preds.shape}")
        print(
            f"Baseline distribution shape: {baseline_dist.shape}, average JS divergence: {np.mean(sample_js_divergences):.4f}")


if __name__ == "__main__":
    shared_globals.init()
    model = load_model("model/cnn_model.keras")
    df = pd.read_csv("data/heart.csv")

    shared_globals.feature_names = df.columns[:-1].tolist()
    shared_globals.feature_data = df[shared_globals.feature_names]

    perturb = CausalFeaturePerturbation(model, df, dataset_name="HeartDisease.csv")
    perturb.run_perturbation()
