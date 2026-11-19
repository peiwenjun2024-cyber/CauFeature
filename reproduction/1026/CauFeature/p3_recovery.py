"""Flat P3 residual interaction recovery.

This module implements the behavior that the active fast30 pipeline actually uses:
P2 core x P1-high residual pairs, ranked by product of P1 contributions, with the
same dynamic threshold semantics. Historical PathShapley-only state is removed.
"""
from itertools import product
import numpy as np
import shared_globals

class ResidualInteractionRecovery:
    def __init__(self, candidate_top_k=200):
        self.candidate_top_k = int(candidate_top_k)
        self.model = shared_globals.model
        self.feature_names = list(shared_globals.feature_names)
        self.contribution = {
            name: float(shared_globals.con_list[i])
            for i, name in enumerate(self.feature_names)
        }
        self.perturbation = shared_globals.perturbation_instance
        if self.perturbation is None:
            raise ValueError("P1 perturbation context is missing")

    def _candidate_pairs(self):
        core = sorted(set(shared_globals.filtered_features or []))
        high = set(shared_globals.high_contrib_features or [])
        residual = sorted((set(self.feature_names) - set(core)) & high)
        ranked = sorted(
            (
                -self.contribution[c] * self.contribution[r],
                c,
                r,
            )
            for c in core
            for r in residual
            if c != r
        )
        pairs = [(c, r) for _, c, r in ranked[: self.candidate_top_k]]
        return core, residual, pairs

    def run(self):
        core, residual, pairs = self._candidate_pairs()
        max_con = max(self.contribution.values()) if self.contribution else 1.0
        max_con = max(max_con, 1e-10)
        recovered = set()
        evaluated = 0

        for core_feature, residual_feature in pairs:
            if residual_feature in recovered:
                continue
            c_single = self.contribution.get(core_feature, 0.0)
            r_single = self.contribution.get(residual_feature, 0.0)
            sum_single = c_single + r_single

            # Current fast30 feature_noise is empty after shared_globals.init().
            # Express its effective behavior directly: efficiency is normalized contribution.
            mean_eff_ratio = (
                (c_single / max_con) + (r_single / max_con)
            ) / 2.0
            dynamic_threshold = 1.0 * (1.0 - 0.4 * mean_eff_ratio)
            required = 1e-6 if sum_single < 1e-9 else sum_single * dynamic_threshold

            significant, observed_max = self.perturbation.joint_effect_exceeds(
                [core_feature, residual_feature],
                required_threshold=required,
                max_combinations=10000,
            )
            evaluated += 1
            if significant:
                recovered.add(residual_feature)

        selected = set(core) | recovered
        final = [name for name in self.feature_names if name in selected]
        if not final:
            raise RuntimeError("P3 produced an empty feature subset")
        shared_globals.filtered_features = final
        shared_globals.to_add = recovered
        return {
            "core_features": core,
            "eligible_residual_features": residual,
            "candidate_pairs": len(pairs),
            "evaluated_pairs": evaluated,
            "recovered_features": sorted(recovered),
            "final_features": final,
        }
