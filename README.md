# PRover
PyTorch code for our EMNLP 2020 paper:

[PRover: Proof Generation for Interpretable Reasoning over Rules](https://arxiv.org/abs/2010.02830)

[Swarnadeep Saha](https://swarnahub.github.io/), [Sayan Ghosh](https://sgdgp.github.io/), [Shashank Srivastava](https://www.ssriva.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Installation
This repository is tested on Python 3.8.3.  
You should install PRover on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Download Dataset
Download the dataset as follows:
```
bash scripts/download_data.sh
```

## Training PRover
PRover can be trained by running the following script:
```
bash scripts/train_prover.sh
```
This will train PRover on the ```depth-5``` dataset. Should you wish to train on any of depth-0, depth-1, etc, change the ```data_dir``` path in the script accordingly.  
The trained model folder will be saved inside ```output``` folder.

## Testing PRover

The trained PRover model can be tested by running the following script:
```
bash scripts/test_prover.sh
```
This will output the QA accuracy, save the node predictions at ```prediction_nodes_dev.lst``` and the predicted edge logits at ```prediction_edge_logits_dev.lst```.

## Running ILP Inference

Once the node predictions and the edge logits are saved, you can run ILP inference to get edge predictions as follows:
```
bash scripts/run_inference.sh
```
This will save the edge predictions inside the model folder.

## Evaluation

Once QA, node and edge predictions are saved, you can compute all metrics (QA accuracy, Node accuracy, Edge accuracy, Proof accuracy and Full accuracy) as follows:
```
bash scripts/get_results.sh
```

## Zero-shot Evaluation on Birds-Electricity
Run the above testing, inference and evaluation scripts to test the depth-5 trained PRover model on the Birds-Electricity dataset by appropriately changing the ```data-dir``` path to ```data/birds-electricity``` in all the scripts and lines 187 and 188 in ```utils.py``` with ```test.jsonl``` and ```meta-test.jsonl```.


## Training PRover on ParaRules dataset
Run the following scripts to train PRover on the ParaRules dataset (following similar steps as before):
```
bash scripts/train_natlang.sh
bash scripts/test_natlang.sh
bash scripts/run_inference_natlang.sh
bash scripts/get_results_natlang.sh
```


## Running Other Ablations
Ablation models from the paper can be run by uncommenting parts of the code (like choosing a particular depth). Please refer to the comments in utils.py for details.

## Trained Models
We also release our trained models on depth-5 dataset and ParaRules dataset [here](https://drive.google.com/file/d/1bvIZMqN2bxw2t1hXbW0WgZkrNWNKkzdC/view?usp=sharing). These contain the respective QA, node and edge predictions and you can reproduce the results from the paper by running the evaluation script.

## Visualizing Proofs
The script to visualize PRover's proof graphs as pdfs is ```evaluation/print_graphs.py```. It takes the usual arguments (data directory, node and prediction files) along with a path to the directoty to save the graphs.

### Citation
```
@inproceedings{saha2020prover,
  title={PRover: Proof Generation for Interpretable Reasoning over Rules},
  author={Saha, Swarnadeep and Ghosh, Sayan and Srivastava, Shashank and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```
