# SS-HSU benchmark

In this repository we implement several methods to perform hyperspectral unmixing (HSU) as well as several self-supervised (SS) methods to train these HSU models.

### Structure

- **models.py** contains the implementation of the different models tested:
    - CNNAEU
    - CNN encoder + linear decoder
    - SWAN
    - Transformer AE
    - NALMU

- **data_augmentation.py** contains several methods to perform positive pair generation for contrastive learning