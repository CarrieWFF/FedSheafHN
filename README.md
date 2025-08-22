## Personalized Subgraph Federated Learning with Sheaf Collaboration

Official Code Repository for the paper - [Personalized Subgraph Federated Learning with Sheaf Collaboration (ECAI2025)](https://arxiv.org/abs/2508.13642).

A non-final version of this work was previously released on arXiv - [FedSheafHN: Personalized Federated Learning on Graph-structured Data](https://arxiv.org/abs/2405.16056).

## Data Generation

Please install [METIS](https://github.com/james77777778/metis_python) for data generation.

Then, run the following commands:

```bash
cd data/generators
python disjoint.py
python overlapping.py
```

## Reproducing Results
To reproduce the main results:
```bash
sh ./scripts/disjoint.sh
sh ./scripts/overlapping.sh
```
Modify the script for different datasets and hyperparameters.

## Reference
Our code is developed based on the following repo:

The neural sheaf model is from: [neural-sheaf-diffusion](https://github.com/twitter-research/neural-sheaf-diffusion/tree/master).

The data generator is from: [FED-PUB](https://github.com/JinheonBaek/FED-PUB).

## Citation
For attribution in academic contexts, please use the bibtex entry below:
```
@misc{liang2025fedsheafhn,
      title={Personalized Subgraph Federated Learning with Sheaf Collaboration}, 
      author={Wenfei Liang and Yanan Zhao and Rui She and Yiming Li and Wee Peng Tay},
      year={2025},
      eprint={2508.13642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.13642}, 
}
```

For reference to the earlier version:

```
@misc{liang2024fedsheafhn,
      title={FedSheafHN: Personalized Federated Learning on Graph-structured Data}, 
      author={Wenfei Liang and Yanan Zhao and Rui She and Yiming Li and Wee Peng Tay},
      year={2024},
      eprint={2405.16056},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
