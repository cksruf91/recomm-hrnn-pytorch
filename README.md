Hierarchical Recurrent Neural Networks Pytorch version
------

Unofficial implementation of [Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks](https://arxiv.org/pdf/1706.04148.pdf)

___work in progress___

## Dataset
* [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/)
```bash
cd {project Dir}/recomm-hrnn-pytorch/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip
```

## Preprocess data
```bash
python preprocess.py -d 10M
```

## Train model
```bash
python train_hrnn.py -d 10M -v 0.0.1
python train_hrnn.py -d BRUNCH -v 0.0.1
```

## Inference model
[TODO]

