import os

import shared_globals
import logging
from tqdm import tqdm
from typing import Dict

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tqdm


class PathShapleyModule:
    """Path Shapley value calculation module (with path length decay functionality)"""

    def __init__(self, log_verbose: bool = False, decay_type: str = "inverse", alpha: float = 0.8):
        self.decay_type = decay_type
        self.alpha = alpha
        self.log_verbose = log_verbose
        self._check_required_globals()

        self.model = shared_globals.model
        self.feature_names = shared_globals.feature_names
        self.target_name = shared_globals.all_names[-1]
        self.feature_data = shared_globals.feature_data
        self.original_preds = shared_globals.original_preds
        self.path_separators = shared_globals.path_separators

        self.node_efficiency = self._compute_node_efficiency()

        self.input_dim = len(self.feature_names)
        self.baseline = shared_globals.baselines
        self.sample_size = self.feature_data.shape[0] if not self.feature_data.empty else 0
        self.max_sample = getattr(shared_globals, 'max_sample', 1000)
        self.cache = {}
        self.baseline_pred = self._init_baseline_pred()

        self.logger = self._init_logger()
        decay_desc = f"1/L (L is path length)" if decay_type == "inverse" else f"α^(L-1) (α={alpha})"

        self._validate_model_input_shape()

    def _init_logger(self) -> logging.Logger:
        """Initialize logger (extract reuse logic into a function)"""
        logger = logging.getLogger("PathShapley")
        logger.setLevel(logging.INFO if self.log_verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger

    def _init_baseline_pred(self) -> float:
        """Initialize baseline prediction value (fix type conversion)"""
        samples = self.feature_data
        input_shape = self.model.input_shape[1:]
        reshaped = samples.values.reshape(-1, *input_shape)

        baseline_preds = self.model.predict(reshaped, verbose=0).flatten()
        return baseline_preds.mean()

    def _validate_model_input_shape(self) -> None:
        if not hasattr(self.model, 'input_shape'):
            if self.log_verbose:
                self.logger.warning("Model has no input_shape attribute, skipping shape validation")
            return
        model_input_dim = self.model.input_shape[1] if len(self.model.input_shape) == 3 else self.model.input_shape[0]
        if self.input_dim != model_input_dim:
            raise ValueError(
                f"Feature count does not match model input: feature count={self.input_dim}, model expects={model_input_dim}")

    def run_redundant_check(self):
        """
        Integrate complete workflow of redundant feature identification, combination evaluation, and validation
        """

        self.get_redundant_features()

        print(
            f"Identified {len(shared_globals.redundant_features)} non-causal features: {shared_globals.redundant_features}")
        print(
            f"Identified {len(shared_globals.high_contrib_features)} high-contribution features: {shared_globals.high_contrib_features}")

        self.run_double_feature_perturbation(threshold=1.2)

    def get_redundant_features(self):
        """
        Subtract features involved in valid paths from global features to get redundant features
        """

        shared_globals.redundant_features = list(
            set(shared_globals.feature_names) - set(
                shared_globals.filtered_features if shared_globals.filtered_features is not None else []))
        return shared_globals.redundant_features

    def run_double_feature_perturbation(self, threshold: float) -> None:
        """
        Perform dual-feature perturbation: verify interaction effects between causal and non-causal subsets, extend causal subset
        :param threshold: Threshold for joint effect to exceed sum of individual effects (e.g., 1.2 means joint effect must exceed 1.2x sum of individuals)
        """

        causal_features = shared_globals.filtered_features

        redundant_set = set(shared_globals.redundant_features)
        high_contrib_set = set(shared_globals.high_contrib_features)

        intersection = redundant_set & high_contrib_set

        non_causal_features = list(intersection)

        print(f"Intersection of non-causal feature subset ∩ high-contribution feature subset: {non_causal_features}")

        if not non_causal_features:
            self.logger.info(
                "Intersection of non-causal feature subset ∩ high-contribution feature subset is empty, no need to perform dual-feature perturbation")
            return

        self.logger.info(
            f"Starting dual-feature perturbation: {len(causal_features)} causal features, {len(non_causal_features)} features in non-causal ∩ high-contribution subset")

        if not hasattr(shared_globals, 'perturbation_instance'):
            raise ValueError(
                "Feature perturbation instance not initialized, please run CausalFeaturePerturbation first")
        perturbation = shared_globals.perturbation_instance
        single_effect = self._get_single_feature_effect()

        feature_pairs = self._heuristic_pair_selection(causal_features, non_causal_features)
        self.logger.info(f"Generated {len(feature_pairs)} dual-feature combinations")

        to_add = set()
        to_add_lock = threading.Lock()
        max_workers = min(os.cpu_count(), len(feature_pairs))

        def process_pair(c_feat, nc_feat):

            with to_add_lock:
                if nc_feat in to_add:
                    return None

            try:
                print(f"\n----- Starting calculation of joint effect for combination ({c_feat}, {nc_feat}) -----")
                joint_effect = perturbation.calculate_joint_contributions([c_feat, nc_feat])
            except Exception as e:

                print(f"\n===== Calculation failed for combination ({c_feat}, {nc_feat}) =====")
                print(f"Error reason: {str(e)}")

                for feat in [c_feat, nc_feat]:
                    if feat in shared_globals.feature_names:
                        idx = shared_globals.feature_names.index(feat)
                        perturb_vals = shared_globals.x_list[idx]
                        print(f"Length of perturbation values for feature {feat}: {len(perturb_vals)}")
                        print(
                            f"First 3 perturbation values for feature {feat}: {perturb_vals[:3] if len(perturb_vals) >= 3 else []}")
                self.logger.warning(f"Failed to calculate joint effect for combination ({c_feat}, {nc_feat}): {e}")
                return None

            c_eff = self.node_efficiency.get(c_feat, 0.0)
            nc_eff = self.node_efficiency.get(nc_feat, 0.0)
            max_eff = max(self.node_efficiency.values()) if self.node_efficiency else 1.0
            mean_eff_ratio = (c_eff + nc_eff) / (2 * max_eff)

            dynamic_threshold = threshold * (1 - 0.4 * mean_eff_ratio)

            c_effect = single_effect.get(c_feat, 0.0)
            nc_effect = single_effect.get(nc_feat, 0.0)
            sum_single = c_effect + nc_effect

            if sum_single < 1e-9:
                if joint_effect > 1e-6:
                    self.logger.info(
                        f"Combination ({c_feat}, {nc_feat}): joint effect {joint_effect:.4f} > sum of individual effects 0, marked as significant")
                    return nc_feat
            else:
                if joint_effect > sum_single * dynamic_threshold:
                    self.logger.info(
                        f"Combination ({c_feat}, {nc_feat}): joint effect {joint_effect:.4f} > sum of individual effects {sum_single:.4f}×{dynamic_threshold}, marked as significant"
                    )
                    return nc_feat
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = {
                executor.submit(process_pair, c_feat, nc_feat): (c_feat, nc_feat)
                for c_feat, nc_feat in feature_pairs
            }

            for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Dual-feature perturbation evaluation (parallel)"
            ):
                result = future.result()
                if result is not None:
                    with to_add_lock:
                        to_add.add(result)

        if to_add:
            extended_causal = list(set(causal_features) | to_add)
            shared_globals.filtered_features = extended_causal
            self.logger.info(
                f"Dual-feature perturbation completed, added {len(to_add)} features to feature selection subset: {sorted(to_add)}, "
                f"updated feature selection subset has {len(extended_causal)} features"
            )
        else:
            self.logger.info(
                "No non-causal-high-contribution features with significant interaction effects found, causal subset remains unchanged")

    def _heuristic_pair_selection(self, causal_features, non_causal_features, top_k=200):
        """Select high-potential feature pairs based on contribution values"""

        causal_contrib = {
            f: shared_globals.con_list[shared_globals.feature_names.index(f)]
            for f in causal_features
        }
        non_causal_contrib = {
            f: shared_globals.con_list[shared_globals.feature_names.index(f)]
            for f in non_causal_features
        }

        pairs = []
        for c_feat in causal_features:
            for nc_feat in non_causal_features:
                score = causal_contrib[c_feat] * non_causal_contrib[nc_feat]
                pairs.append((-score, c_feat, nc_feat))

        pairs.sort()
        selected = [(c, nc) for (_, c, nc) in pairs[:top_k]]
        return selected

    def _get_single_feature_effect(self) -> Dict[str, float]:
        """Generate mapping from feature names to single-feature effect values (reuse con_list)"""
        if not hasattr(shared_globals, 'con_list') or not hasattr(shared_globals, 'feature_names'):
            raise ValueError("Missing feature contribution data, please run the feature perturbation module first")
        return {
            feat: shared_globals.con_list[i]
            for i, feat in enumerate(shared_globals.feature_names)
        }

    def _check_required_globals(self) -> None:
        required = [
            'model', 'feature_names', 'baselines', 'all_names',
            'path_separators', 'tolerance', 'random_seed',
            'feature_data', 'original_preds'
        ]
        missing = [var for var in required if not hasattr(shared_globals, var)]
        if missing:
            raise ValueError(f"Missing global variables, cannot initialize module: {missing}")

    def _compute_node_efficiency(self) -> Dict[str, float]:
        """Calculate node efficiency (reuse global variables)"""
        epsilon = 1e-10
        efficiency = {feat: shared_globals.con_list[i] / (shared_globals.feature_noise.get(feat, 0.0) + epsilon)
                      for i, feat in enumerate(self.feature_names)}
        efficiency[self.target_name] = 1.0
        return efficiency
