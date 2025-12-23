# SS-HSU benchmark

This repository implements several methods to perform hyperspectral unmixing (HSU) as well as several self-supervised (SS) methods to train these HSU models.

### Structure

The **/src** folder contains the following files:

- **/models**:

    `models.py` contains the implementation of the different models tested:
    - MLP Autoencoder
    - CNN encoder + linear decoder 
    - CNNAEU [[1]](#1)
    - Deep Transformer AE [[2]](#2)
    - NALMU [[3]](#3)
    - RALMU [[3]](#3)
  
    `transformer.py` contains the source code for the Transformer AE model.

    `foundation_models.py` contains the *FoundationModel* class, implementing the following foundation models as feature extractor used for hsu: *SpectralEarth* [[4]](#4), *SpectralGPT* [[5]](#5), *DOFA* [[6]](#6), *HyperFree* [[7]](#7), *HyperSL* [[8]](#8), *HyperSIGMA* [[9]](#9).

- **/training**:

    `data_augmentation.py` contains some data augmentation functions for the contrastive learning method.
    
    `self-supervision.py` contains the implementation of different self-supervised training methods:
    - DIP [[10]](#10)
    - Two Stages network [[11]](#11)
    - Synthetic Training Dataset Generation [[12]](#12)
    - Contrastive Learning [[13]](#13)

- **/utils**:
  
     `extractor.py` contains the VCA and FCLS method for a simple unmixing method.

     `utils.py` contains some utils functions and classes such as dataset creation, loss functions, model saving, results visualization and metrics computation.

## References
<a id="1">[1]</a> 
Pallson et al. “Convolutional Autoencoder for Spectral–Spatial  Hyperspectral Unmixing”. In : IEEE TGRS 2021
url : [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9096565](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9096565)

<a id="2">[2]</a>
Gosh et al. "Deep Hyperspectral Unmixing using Transformer Network". In: IEEE TGRS 2022
url: [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9848995](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9848995)

<a id="3">[3]</a> 
Christophe Kervazo et Jérémy Cohen. “Unrolled Multiplicative Updates for Nonnegative Matrix Factorization applied to Hyperspectral Unmixing”. In : In prep.

<a id="4">[4]</a>
Hong et al. "SpectralGPT: Spectral Remote Sensing Foundation Model". In: IEEE TPAMI 2024
url: [https://arxiv.org/pdf/2311.07113](https://arxiv.org/pdf/2311.07113)

<a id="5">[5]</a>
Braham et al. "SpectralEarth: Training Hyperspectral Foundation Models at Scale". In: arXiv preprint 2024
url: [https://arxiv.org/pdf/2408.08447](https://arxiv.org/pdf/2408.08447)

<a id="6">[6]</a>
Xiong et al. "Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation". In: arXiv preprint 2024
url: [https://arxiv.org/pdf/2403.15356](https://arxiv.org/pdf/2403.15356)

<a id="7">[7]</a>
J. Li, Y. Liu et al. "HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery". In: CVPR 2025
url: [https://arxiv.org/pdf/2503.21841](https://arxiv.org/pdf/2503.21841)

<a id="8">[8]</a>
Kong et al. "HyperSL: A Spectral Foundation Model for Hyperspectral Image Interpretation". In: IEEE TGRS 2025
url: [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10981753](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10981753)

<a id="9">[9]</a>
Wang et al. "HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model". In: IEEE GRSL 2022
url: [https://ieeexplore.ieee.org/document/9734031/](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10949864)

<a id="10">[10]</a>
Rasti et al. "UnDIP: Hyperspectral Unmixing Using Deep Image Prior". In: IEEE TPAMI 2025
url: [https://ieeexplore.ieee.org/abstract/document/9392110](https://ieeexplore.ieee.org/abstract/document/9392110)

<a id="11">[11]</a>
S. S et al. "A Practical Approach for Hyperspectral Unmixing Using Deep Learning". In: IEEE GRSL 2022
url: [https://ieeexplore.ieee.org/document/9610077/](https://ieeexplore.ieee.org/document/9610077/)

<a id="12">[12]</a>
Hadjeres et al. "Generating Synthetic Data to Train a Deep Unrolled Network for Hyperspectral Unmixing". In: EUSIPCO 2024
url: [https://ieeexplore.ieee.org/document/10714958/](https://ieeexplore.ieee.org/document/10714958/)

<a id="13">[13]</a>
Zhao et al. "Hyperspectral Image Classification With Contrastive Self-Supervised Learning Under Limited Labeled Samples". In: IEEE GRSL 2022
url: [https://ieeexplore.ieee.org/document/9734031/](https://ieeexplore.ieee.org/document/9734031/)
