# RuleReasoning
Our code uses python 3.8. Check requirements.txt for dependencies.

# Training A Model

To train a PRover model, use scripts/train_prover.sh.
Use the appropriate arguments for the model paths, etc.

# Testing A Model

To test a PRover model, use scripts/test_prover.sh.
This outputs the QA accuracy, saves the node predictions and the predicted edge logits.

# Running ILP Inference

To run ILP Inference for generating edges, use scripts/run_inference.sh
This outputs the edge predictions.

# Computing all metrics

To get QA accuracy and all proof-related metrics, use scripts/get_results.sh.

# Running other experiments
The ParaRules experiments can be run using the scripts/train_natlang.sh script and following similar steps as previously listed.

Ablation models can be run by uncommenting parts of the code (like choosing a particular depth). Please refer to the comments in utils.py for details.