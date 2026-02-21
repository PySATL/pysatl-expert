import json


class Report:
    """
    Data Object representing the final verdict of the identification process.

    Attributes:
        distribution_name (str): The name of the winning distribution.
        confidence (float): Confidence level derived from bootstrap voting (0.0 to 1.0).
        all_scores (dict): Raw Goodness-of-Fit scores for the original sample.
        parameters (dict, optional): Fitted parameters of the winning distribution.
        final_ranks (dict, optional): Voting results or penalty scores for all candidates.
    """

    def __init__(
        self, distribution_name, confidence, all_scores, parameters=None, final_ranks=None
    ):
        self.distribution_name = distribution_name
        self.confidence = confidence
        self.all_scores = all_scores
        self.parameters = parameters
        self.final_ranks = final_ranks

    def __str__(self):
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

    def __repr__(self):
        return self.__str__()
