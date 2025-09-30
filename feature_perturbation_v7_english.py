import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
import os

import shared_globals

plt.rcParams['font.sans-serif'] = ['SimHei']  # Keep Chinese font support
plt.rcParams['axes.unicode_minus'] = False

import time
from typing import List, Tuple
from tensorflow.keras.models import load_model
from scipy.stats import entropy


class CausalFeaturePerturbation:
    """Feature perturbation module adapted to CausalFeatureX framework (parallel computing optimized version)"""

    def __init__(self, model: tf.keras.Model, data: pd.DataFrame):
        self.model = model
        self.data = data
        self.sample_size = data.shape[0]
        self.baseline = self._calculate_baseline()
        self.baseline_np = self.baseline.values.astype(np.float32).copy()
        self.n_features = len(shared_globals.feature_names)

        self.baseline_reshaped = self.baseline_np.reshape(-1, self.n_features, 1)

        self.baseline_pred = self.model.predict(self.baseline_reshaped, verbose=0)

        self.baseline_pred_dist = self.baseline_pred[0]

        self.feature_values = {}
        for col in shared_globals.feature_names:
            try:
                self.feature_values[col] = sorted(pd.to_numeric(data[col], errors='coerce').dropna().unique().tolist())
            except:
                self.feature_values[col] = []

        shared_globals.baselines = self.baseline
        shared_globals.baseline_preds = self.baseline_pred

    def _calculate_baseline(self):
        """Calculate baseline values"""
        baseline = pd.Series(index=shared_globals.feature_names, dtype=np.float32)
        for feature in shared_globals.feature_names:
            baseline[feature] = shared_globals.feature_data[feature].mean()
        return pd.DataFrame([baseline])

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

    def disturbance_range(self, min_list: List[float], max_list: List[float],
                          count_list: List[int], range_list: List[List[float]]) -> List[List[float]]:
        """Core perturbation value generation function"""
        x_list = []
        HIGH_DIVERSITY_THRESHOLD = 0.9
        LARGE_DIFF_THRESHOLD = 10000
        NUM_PARTITIONS = 100
        SPARSITY_THRESHOLD = 0.001

        for j, col in tqdm(enumerate(shared_globals.feature_names),
                           desc="Generating feature perturbation values",
                           total=len(shared_globals.feature_names)):
            min_val, max_val = min_list[j], max_list[j]
            unique_count, val_range = count_list[j], range_list[j]

            if min_val > max_val:
                x_list.append([])
                continue

            diff = max_val - min_val
            diversity_ratio = unique_count / self.sample_size if self.sample_size > 0 else 0

            if diff >= LARGE_DIFF_THRESHOLD:
                if diversity_ratio >= HIGH_DIVERSITY_THRESHOLD:

                    perturb_values = np.linspace(min_val, max_val, NUM_PARTITIONS + 1, dtype=np.float32)
                    x_list.append(np.round(perturb_values, 6).tolist())
                else:

                    if diversity_ratio < SPARSITY_THRESHOLD:

                        x_list.append(sorted(val_range))
                    else:
                        x_list.append(list(range(int(min_val), int(max_val) + 1)))


            else:
                if diff <= 1 and any(not float(v).is_integer() for v in val_range):
                    perturb_values = np.linspace(min_val, max_val, unique_count, dtype=np.float32)
                    x_list.append(np.round(perturb_values, 6).tolist())
                else:
                    x_list.append(list(range(int(min_val), int(max_val) + 1)))

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

            preds = self.model.predict(batch_tensor, verbose=0)

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
        """Parallel calculation of feature contributions (using JS divergence)"""
        y_list = [[] for _ in shared_globals.feature_names]
        con_list = [0.0 for _ in shared_globals.feature_names]
        start = time.perf_counter()

        features = shared_globals.feature_names
        max_workers = min(8, os.cpu_count())
        print(f"Using {max_workers} threads for parallel feature contribution calculation...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for j, col in enumerate(features):
                values = x_list[j]
                if values:
                    futures.append(executor.submit(
                        self._process_single_feature,
                        j, col, values
                    ))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating feature contributions"):
                j, y, con = future.result()
                y_list[j] = y
                con_list[j] = con

        end = time.perf_counter()
        print(f"Total time for feature perturbation experiments: {end - start:.2f} seconds")
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
        preds = self.model.predict(batch_input_reshaped, verbose=0)

        max_js = 0.0
        for dist in preds:
            js_div = self._js_divergence(dist, self.baseline_pred_dist)
            if js_div > max_js:
                max_js = js_div
        return max_js

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

    def feature_important(self, con_list: List[float]):
        """Feature importance analysis (using JS divergence results)"""
        total_con = sum(con_list)
        if total_con == 0:
            return

        con_list = [max(0.0, con) for con in con_list]
        total_con = sum(con_list)

        top_indices = np.argsort(con_list)[-10:][::-1]
        merged_con = [con_list[i] for i in top_indices]
        merged_names = [shared_globals.feature_names[i] for i in top_indices]
        other_con = total_con - sum(merged_con)
        other_con = max(0.0, other_con)

        merged_con.append(other_con)
        merged_names.append("others")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(merged_names, merged_con, alpha=0.6)
        axs[0].tick_params(axis='x', rotation=70)
        axs[1].pie(merged_con, labels=merged_names, autopct='%1.1f%%')
        plt.tight_layout()
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

    perturb = CausalFeaturePerturbation(model, df)
    perturb.run_perturbation()
