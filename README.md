# MoSE

Code base for ["Homomorphism Counts as Structural Encodings for Graph Learning"](https://arxiv.org/abs/2410.18676) (also presented under the title "Homomorphism Counts as Structural Encodings for Molecular Property Prediction" at the NeurIPS 2024 AIDrugX Workshop).

Our repository combines the [GraphGPS](https://github.com/rampasek/GraphGPS.git) repository from ["Recipe for a General, Powerful, Scalable Graph Transformer"](https://arxiv.org/abs/2205.12454), the [hombasis-gnn](https://github.com/ejin700/hombasis-gnn.git) repository from ["Homomorphism Counts for Graph Neural Networks"](https://arxiv.org/abs/2402.08595), and the [GRIT](https://github.com/LiamMa/GRIT) repository from ["Graph Inductive Biases in Transformers without Message Passing"](https://arxiv.org/abs/2305.17589).

## Python Virtual Enviroment Setup

```bash
conda create --name HomEnv python=3.12.3
conda activate HomEnv

pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==2.5.3

pip install ogb easydict pyyaml neptune wandb yacs

pip install opt_einsum

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

pip install tensorboardX
pip install performer-pytorch
pip install torchmetrics
pip install numpy==1.26.4

pip install dill
```

## Zinc Data Setup
To set up the data necessary for Spm experiments on ZINC, unzip the `hombasis-gt/hombasis-bench/data/zinc-data.zip` file into the `hombasis-gt/hombasis-bench/data` directory. 

## QM9 Data Setup
To set up the data necessary for Hom experiments on QM9, unzip the file `hombasis-gt/qm9/data/QM9/v5_homcounts.zip`, and move the resulting files (`test_homcounts.json`, `train_homcounts.json`, `valid_homcounts.json`) into the `hombasis-gt/qm9/data/QM9` directory. Then, run the python script `hombasis-gt/qm9/data_GraphGym_QM9/save_qm9_hc.py` in order to process the count-enhanced QM9 dataset (will be saved as `datasets/QM9-GraphHC/processed/joined.pt`). It may take a few minutes for `save_qm9_hc.py` to run.

## Synth Data Setup
To set up our synthetic (random graph) dataset, run the script `hombasis-gt/synth/save_synth_dataset.py` (this will save the homomorphism count enhanced datasets to `datasets/SYNTH-All5/processed` and `datasets/SYNTH-Spasm/processed`). 

## Running Experiments
To run an experiment, set up a `configuration.yaml` file containing the model hyperparameters and experimental setup such as those given in the `GraphGPS/configs/` directory. Then, run:

```bash
python GraphGPS/main.py --cfg "/path_to/configuration.yaml" --repeat 1 wandb.use True
```

For example, replicate the best result for ZINC GPS+Spasm by running:

```bash
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 0
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 14
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 48
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 96
```

To run experiments with GRIT, use the `GRIT/main.py` script instead of `GraphGPS/main.py`, and use config files given in `GRIT/configs/`.

**Caution**: The model hyperparameters in the experimental details section of our [ArXiv](https://arxiv.org/abs/2410.18676) posting has some mistakes. We will post a revised version with the correct hyperparameters (and a few other exciting experimental results on additional datasets!) as soon as we can :blush:. For now, please reference the "config" directories in this repository for hyperparameters details.