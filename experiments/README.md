# Experiments

This folder contains the python code for evaluating our results. In case you need to look up some details, here is an overview over all the different files:

1. **mona_analysis_add_label_columns.py**: A helper function to expand an existing result-csv. It calculates the label for each evaluated scenario and then adds match@k by comparing the label to the LLM-ranking.
2. **mona_analysis.py**: For a MONA run, calculate all the important metrics.
3. **rule_monitoring_config_assembled.py**: 
4. **rule_monitoring_config_separate.py**: 
5. **sandra_analysis_fail-safe.py**: For a highwayenv run, this file computes the share of decision-making steps where no verifiable action was found.
6. **sandra_analysis_ratio.py**: For all highwayenv runs, compute the frequency with which the LLM selected any of the available actions. Low-level actions are the actual (lon, lat) tuple which the LLM generated. High-level actions are the highwayenv action the tuple got mapped to.
7. **sandra_analysis_xml.py**: For all highwayenv runs, compute the standard metrics, like travelled distance or success rates.
8. **sandra_analysis_xml_rule.py**: Same as above but also computes rule-compliance.
9. **data**: Contains the data from MONA experiments. You can use it to recompute the metrics.
