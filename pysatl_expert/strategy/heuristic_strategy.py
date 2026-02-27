from collections import Counter

from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.report import Report


class HeuristicStrategy(AbstractStrategy):
    """
    Implements a rule-based heuristic strategy for distribution selection.

    This strategy aggregates diverse Goodness-of-Fit (GoF) statistics into a single
    scalar 'penalty' metric to determine the best candidate. Since different
    statistical tests operate on different scales (e.g., p-values vs. statistic values),
    this class applies normalization factors to make them comparable.

    The decision logic incorporates:
    1.  **Normalization**: Scaling deviations (e.g., Gini index) to match the
        magnitude of standard statistics (e.g., Anderson-Darling).
    2.  **Occam's Razor**: Applying a complexity penalty to multi-parameter
        distributions (e.g., Weibull) to prefer simpler models (e.g., Exponential)
        when fits are comparable.
    3.  **Inversion**: converting maximization metrics (like Shapiro-Wilk p-value)
        into minimization penalties.

    Attributes:
        GINI_NORMALIZATION_SCALE (float): Scaling factor for Gini index deviation.
            The index theoretical variance is small (< 0.05), so it is scaled up
            to match the magnitude of AD statistics (~1.0).
        MODEL_COMPLEXITY_PENALTY (float): Additive penalty for the Weibull
            distribution (2 parameters) to prevent overfitting against the
            Exponential distribution (1 parameter).
        SHAPIRO_INVERSE_SCALE (float): Factor to convert (1 - W) into a penalty score.
    """

    GINI_NORMALIZATION_SCALE = 20.0
    MODEL_COMPLEXITY_PENALTY = 0.2
    SHAPIRO_INVERSE_SCALE = 30.0

    def _calculate_penalty(self, dist_name: str, scores: dict, debug: bool = False) -> float:
        """
        Normalizes distribution scores into a unified penalty scale (lower is better).
        """
        name = dist_name.lower()
        penalty = 100.0
        debug_info = ""

        if "normal" in name:
            if "anderson_darling" in scores:
                penalty = scores["anderson_darling"]
                debug_info = f"AD={penalty:.4f}"
            elif "shapiro_wilk" in scores:
                w_stat = scores["shapiro_wilk"]
                penalty = (1.0 - w_stat) * self.SHAPIRO_INVERSE_SCALE
                debug_info = f"Shapiro (1-{w_stat:.4f})*{self.SHAPIRO_INVERSE_SCALE}"

        elif "expon" in name:
            if "gini_index" in scores:
                raw_val = scores["gini_index"]
                diff = abs(raw_val - 0.5)
                penalty = diff * self.GINI_NORMALIZATION_SCALE
                debug_info = f"Gini |{raw_val:.4f}-0.5|*{self.GINI_NORMALIZATION_SCALE}"
            elif "moran_test" in scores:
                penalty = scores["moran_test"]
                debug_info = f"Moran={penalty:.4f}"

        elif "weibull" in name:
            base_penalty = 100.0
            if "anderson_darling" in scores:
                base_penalty = scores["anderson_darling"]
                debug_info = f"AD={base_penalty:.4f}"
            elif "tiku_singh" in scores:
                ts_stat = scores["tiku_singh"]
                base_penalty = (1.0 - ts_stat) * 10.0
                debug_info = f"TikuSingh (1-{ts_stat:.4f})*10"

            penalty = base_penalty + self.MODEL_COMPLEXITY_PENALTY
            debug_info += f" + Penalty({self.MODEL_COMPLEXITY_PENALTY})"

        if debug:
            print(f"[DEBUG] {dist_name:<12} | {debug_info:<30} | Final: {penalty:.4f}")

        return penalty

    def _choose_winner_from_fv(self, fv: FeatureVector, debug: bool = False) -> str:
        """
        Determines the winner for a single feature vector based on minimum penalty.
        """
        scores = fv.candidates_scores
        active_dists = [d for d in scores.keys() if scores[d]]

        if not active_dists:
            return "None"

        penalties = {}
        if debug:
            print("-" * 60)

        for name in active_dists:
            penalties[name] = self._calculate_penalty(name, scores[name], debug)

        if debug:
            print("-" * 60)

        sorted_res = sorted(penalties.items(), key=lambda x: x[1])
        winner, min_penalty = sorted_res[0]

        return winner

    def predict_report(self, base_fv: FeatureVector, bootstrap_fvs=None) -> Report:
        """
        Generates the final identification report using original data and bootstrap voting.
        """
        base_scores = base_fv.candidates_scores

        ranks = {}
        active_dists = [d for d in base_scores.keys() if base_scores[d]]
        for name in active_dists:
            ranks[name] = self._calculate_penalty(name, base_scores[name], debug=False)

        if not bootstrap_fvs:
            winner = self._choose_winner_from_fv(base_fv, debug=False)
            return Report(winner, 0.0, base_scores, final_ranks=ranks)

        votes = []
        for fv in bootstrap_fvs:
            w = self._choose_winner_from_fv(fv, debug=False)
            if w != "None":
                votes.append(w)

        if not votes:
            return Report("None", 0.0, base_scores)

        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        confidence = count / len(votes)

        return Report(
            distribution_name=winner,
            confidence=round(confidence, 2),
            all_scores=base_scores,
            final_ranks=dict(vote_counts),
        )
