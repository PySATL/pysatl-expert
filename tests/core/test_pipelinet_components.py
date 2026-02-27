from unittest.mock import MagicMock

from pysatl_expert.core.pipeline_components import PipelineComponents


def test_pipeline_components_init():
    mock_distributions = [MagicMock(), MagicMock()]
    mock_selector = MagicMock()
    mock_strategy = MagicMock()
    mock_extractor = MagicMock()

    components = PipelineComponents(
        distributions=mock_distributions,
        criterion_selector=mock_selector,
        strategy=mock_strategy,
        feature_extractor=mock_extractor,
    )

    assert components.distributions == mock_distributions
    assert components.criterion_selector == mock_selector
    assert components.strategy == mock_strategy
    assert components.feature_extractor == mock_extractor
    assert len(components.distributions) == 2
