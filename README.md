# SS-HSU benchmark

This repository implements several methods to perform hyperspectral unmixing (HSU) as well as several self-supervised (SS) methods to train these HSU models.

### Structure

The **/src** folder contains the following files:

- **/models**:

    `models.py` contains the implementation of the different models tested:
    - CNNAEU [[1]](#1)
    - CNN encoder + linear decoder 
    - Transformer AE [[2]](#2)
    - NALMU [[3]](#3)
    - RALMU [[3]](#3)
  
    `transformer.py` contains the source code for the Transformer AE model

- **/data**:

    Contains the dataset creation helper function `data_utils.py`

- **/training**:

    `data_augmentation.py` contains some data augmentation functions for the contrastive learning method
    
    `self-supervision.py` contains the implementation of different self-supervised training methods:
    - DIP [[4]](#4)
    - Two Stages network [[5]](#5)
    - Synthetic Training Dataset Generation [[6]](#6)
    - Contrastive Learning [[7]](#7)

- **/utils**:
     `extractor.py` contains the VCA and FCLS method for a simple unmixing method

## References
<a id="1">[1]</a> 
Pallson et al. “Convolutional Autoencoder for Spectral–Spatial  Hyperspectral Unmixing”. In : IEEE TGRS 2021
url : [https://hal.science/hal-04736884](https://ieeexplore.ieee.org/document/9096565/)

<a id="2">[2]</a>
Gosh et al. "Deep Hyperspectral Unmixing using Transformer Network". In: IEEE TGRS 2022
url: http://arxiv.org/abs/2203.17076

<a id="3">[3]</a> 
Christophe Kervazo et Jérémy Cohen. “Unrolled Multiplicative Updates for Nonnegative Matrix Factorization applied to Hyperspectral Unmixing”. In : In prep.

<a id="4">[4]</a>
Rasti et al. "UnDIP: Hyperspectral Unmixing Using Deep Image Prior". In: IEEE TGRS 2022
url: [http://arxiv.org/abs/2203.17076](https://ieeexplore.ieee.org/document/9392110/)

<a id="5">[5]</a>
S. S et al. "A Practical Approach for Hyperspectral Unmixing Using Deep Learning". In: IEEE GRSL 2022
url: [http://arxiv.org/abs/2203.17076](https://ieeexplore.ieee.org/document/9610077/)

<a id="6">[6]</a>
Hadjeres et al. "Generating Synthetic Data to Train a Deep Unrolled Network for Hyperspectral Unmixing". In: EUSIPCO 2024
url: [http://arxiv.org/abs/2203.17076](https://ieeexplore.ieee.org/document/10714958/)

<a id="7">[7]</a>
Zhao et al. "Hyperspectral Image Classification With Contrastive Self-Supervised Learning Under Limited Labeled Samples". In: IEEE GRSL 2022
url: [http://arxiv.org/abs/2203.17076](https://ieeexplore.ieee.org/document/9734031/)
