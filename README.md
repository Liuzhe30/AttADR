# AttADR
A ddi side-effect classfication tool using attention-based deep learning network.

<p align="center"><img width="100%" src="AttADR.png" /></p>

## Quick Start

### Requirements
- Python 3.6+
- Tensoflow == 2.7.0

### Download AttADR
```shell
git-lfs clone https://github.com/Liuzhe30/AttADR
```

### Dataset Preparation
In this project, the three datasets corresponding to the three classification tasks are [dataset_task1](https://github.com/Liuzhe30/AttADR/blob/main/data/pair_label_task1.npy)
, [dataset_task2](https://github.com/Liuzhe30/AttADR/blob/main/data/pair_label_test_task2.npy)
 and [dataset_task3](https://github.com/Liuzhe30/AttADR/blob/main/data/pair_label_test_task3.npy) 
 respectively.

### Evaluate model
```shell
python3 source/evaluate_AttADR.py
```

## Contributing to the project
Any pull requests or issues are welcome.

## Progress
- [x] README for running AttADR.