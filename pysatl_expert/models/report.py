import json


class Report:
    """Represents the final verdict and statistical breakdown of distribution identification.

    Attributes:
        distribution_name (str): Name of the identified winning distribution.
        confidence (float): Confidence level score (0.0 to 1.0).
        all_scores (dict): Raw GoF scores for all candidate distributions.
        parameters (dict | None): Fitted parameters of the winning distribution.
        final_ranks (dict | None): Voting or rank breakdown across candidate distributions.
    """

    def __init__(
        self,
        distribution_name: str,
        confidence: float,
        all_scores: dict,
        parameters: dict | None = None,
        final_ranks: dict | None = None,
        confidence_kind: str = "model_probability",
        model_confidence: float | None = None,
        bootstrap_stability: float | None = None,
        model_ranks: dict | None = None,
        stage1_scores: dict | None = None,
        stage2_scores: dict | None = None,
        stage1_features: dict | None = None,
        stage2_features: dict | None = None,
        candidate_parameters: dict | None = None,
        bootstrap_ranks: dict | None = None,
        bootstrap_requested: int | None = None,
        bootstrap_successful: int = 0,
        bootstrap_errors: list[str] | None = None,
        sample_statistics: dict[str, float | int] | None = None,
    ):
        """Initialize the identification report.

        Args:
            distribution_name (str): Name of the winning distribution.
            confidence (float): Calculated confidence level (0.0 to 1.0).
            all_scores (dict): Map of all raw GoF criterion scores.
            parameters (dict | None): Estimated parameter dictionary of the winner.
            final_ranks (dict | None): Voting scores or ranks for all candidate models.
            confidence_kind (str): Meaning of confidence: model probability or
                bootstrap stability.
            model_confidence (float | None): ML model score on the base sample;
                not a calibrated probability of the hypothesis being true.
            bootstrap_stability (float | None): Bootstrap vote share for the winner.
            model_ranks (dict | None): Base-sample overall scores before bootstrap voting.
            stage1_scores (dict | None): Family-to-score map from the Stage 1 forest.
            stage2_scores (dict | None): Family-to-distribution conditional score maps.
            stage1_features (dict | None): Selected feature names and actual raw inputs.
            stage2_features (dict | None): Selected raw inputs grouped by family.
            candidate_parameters (dict | None): Successfully fitted candidate parameters.
                Missing candidates do not imply any specific reason for failure.
            bootstrap_ranks (dict | None): Per-class top-1 vote shares on valid resamples;
                independent of the base-sample ranking.
            bootstrap_requested (int | None): Requested repeats, or None if unknown.
            bootstrap_successful (int): Number of valid repeats used for vote shares.
            bootstrap_errors (list[str] | None): Logged failures of resample evaluations.
                Does not include individual criterion warnings within valid resamples.
            sample_statistics (dict[str, float | int] | None): Descriptive statistics
                calculated for the original sample before classification.
        """
        self.distribution_name = distribution_name
        self.confidence = confidence
        self.all_scores = all_scores
        self.parameters = parameters
        self.final_ranks = final_ranks
        # Optional evidence refers to the original sample, not bootstrap votes.
        # Feature mappings contain the actual input values of each selected forest.
        self.model_ranks = dict(model_ranks or {})
        self.stage1_scores = dict(stage1_scores or {})
        self.stage2_scores = dict(stage2_scores or {})
        self.stage1_features = dict(stage1_features or {})
        self.stage2_features = dict(stage2_features or {})
        self.candidate_parameters = dict(candidate_parameters or {})
        self.bootstrap_ranks = dict(bootstrap_ranks or {})
        self.bootstrap_requested = bootstrap_requested
        self.bootstrap_successful = bootstrap_successful
        self.bootstrap_errors = list(bootstrap_errors or [])
        self.sample_statistics = dict(sample_statistics or {})
        self.confidence_kind = confidence_kind
        self.model_confidence = (
            model_confidence
            if model_confidence is not None
            else (confidence if confidence_kind == "model_probability" else None)
        )
        self.bootstrap_stability = (
            bootstrap_stability
            if bootstrap_stability is not None
            else (confidence if confidence_kind == "bootstrap_stability" else None)
        )

    def __str__(self) -> str:
        """Return a human-readable text summary of the identification report.

        Returns:
            str: Formatted multi-line report string.
        """

        def safe_serialize(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        try:
            scores_str = json.dumps(self.all_scores, default=safe_serialize, indent=4)
        except (TypeError, ValueError):
            scores_str = str(self.all_scores)

        lines = [
            "--- Identification Report ---",
            f"Winner:      {self.distribution_name}",
            f"Confidence:  {self.confidence}",
            f"Confidence kind: {self.confidence_kind}",
        ]
        if self.model_confidence is not None:
            lines.append(f"Model confidence:    {self.model_confidence}")
        if self.bootstrap_stability is not None:
            lines.append(f"Bootstrap stability: {self.bootstrap_stability}")
        if self.bootstrap_requested or self.bootstrap_successful:
            lines.append(f"Bootstrap successful: {self.bootstrap_successful}")
            lines.append(f"Bootstrap requested: {self.bootstrap_requested}")
            lines.append(f"Bootstrap top-1 shares: {self.bootstrap_ranks}")
        lines.extend(
            [
                f"Parameters:  {self.parameters}",
                f"Votes/Ranks: {self.final_ranks}",
                f"Detailed Scores:\n{scores_str}\n",
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()
