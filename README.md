# AdaFFed: Adaptive Sample Reweighting for Fair Federated Learning

This repository is built based on PyTorch, containing the code for the manuscript.

## Dataset

- Compas
  
  - Download 
  
  - https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
  
  - Place it in the folder: data/compas/raw_data

- Adult
  
  - Download the following files:
  
  - https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
  
  - Place it in the folder: data/adult/raw_data

- CelebA
  
  - Download files from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    
    - img_align_celeba.7z
    
    - identity_CelebA.txt
    
    - list_attr_celeba.txt
  
  - unzip img_align_celeba.7z, place them in the folder: data/celeba/raw_data

## Training

**Run main.py to train the model.**

Example:

```
python main.py --algorithm fedavg --data compas --num_round 50 --local_lr 0.001

python main.py --algorithm adaffed --data compas --client 2 --a_value 5 --c_value 10 
```

See comments in options.py for more paprameters.
