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
    ):
        """Initialize the identification report.

        Args:
            distribution_name (str): Name of the winning distribution.
            confidence (float): Calculated confidence level (0.0 to 1.0).
            all_scores (dict): Map of all raw GoF criterion scores.
            parameters (dict | None): Estimated parameter dictionary of the winner.
            final_ranks (dict | None): Voting scores or ranks for all candidate models.
        """
        self.distribution_name = distribution_name
        self.confidence = confidence
        self.all_scores = all_scores
        self.parameters = parameters
        self.final_ranks = final_ranks

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

        return (
            f"--- Identification Report ---\n"
            f"Winner:      {self.distribution_name}\n"
            f"Confidence:  {self.confidence}\n"
            f"Parameters:  {self.parameters}\n"
            f"Votes/Ranks: {self.final_ranks}\n"
            f"Detailed Scores:\n{scores_str}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()
