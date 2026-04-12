"""Distilabel synthetic data pipeline integration."""

from slm_auto_config.synthetic.classification_sdg import (
    ClassificationSDGConfig,
    run_classification_sdg,
)


def generate_synthetic_data(config: ClassificationSDGConfig) -> list[dict[str, str]]:
    """Run the classification Distilabel pipeline and persist synthetic data."""
    return run_classification_sdg(config)
