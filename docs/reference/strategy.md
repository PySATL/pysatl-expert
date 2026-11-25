# RF method adapter

`MLStrategy` is the adapter for the hierarchical Random Forest artifact. It maps an `AnalysisResult` to the feature names and order in a verified model schema. This is a special ready-method integration; it does not define the general Expert API or train a model.

The command line and `recommend_ml` require a trusted model file with an adjacent manifest. The manifest is checked before deserialization and analysis. The package does not include a trained bundle; any supplied bundle must match the installed `pysatl-criterion` revision (`v1.0.0`, commit `3685d7e746fd7c175e18dc9b113f1db0d4fe3455`) and exact feature schema.

Scores are uncalibrated model outputs, not probabilities that a statistical hypothesis is true. A matching feature name alone does not prove that a statistic has the semantics expected by an artifact. See [trusted model artifacts](models.md#trusted-model-artifacts) and [statistics and procedure results](criteria.md).

::: pysatl_expert.strategy.ml_strategy
