# Fuzzy-Rank-Ensemble-Torch

[![GitHub Release](https://img.shields.io/github/v/release/Digiratory/fuzzy-rank-ensemble-torch)](https://github.com/Digiratory/fuzzy-rank-ensemble-torch/releases)
[![GitHub License](https://img.shields.io/github/license/Digiratory/fuzzy-rank-ensemble-torch)](https://github.com/Digiratory/fuzzy-rank-ensemble-torch/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fuzzy-rank-ensemble-torch)](https://pypi.org/project/fuzzy-rank-ensemble-torch/)

The torch based implementation of a Fuzzy Rank-based Ensemble compatible with Monai framework and segmentation purpose.

The code is based on our papers:

* "[A fuzzy rank-based ensemble of CNN models for MRI segmentation](https://www.sciencedirect.com/science/article/abs/pii/S1746809424014009)" published in Elseiver Biomedical Signal Processing and Control journal.
* "[A Fuzzy Rank-based Ensemble of CNN Models for Classification of Cervical Cytology](https://www.nature.com/articles/s41598-021-93783-8)" published in Nature-Scientific Reports journal.

## Installation

You can install the genereal purpose package from PyPI:

```bash
pip install fuzzy-rank-ensemble-torch
```

Also, you can install this package with optional dependencies:

```bash
pip install fuzzy-rank-ensemble-torch[monai]
```

Finally, you can also install the package directly from GitHub:

```bash
pip install git+https://github.com/Digiratory/fuzzy-rank-ensemble-torch.git
```

## Usage

### Pure Torch

```python
    import torch
    from fuzzy_rank_ensemble_torch import fuzzy_rank_ensemble

    # Tensor initialization, here we use a dummy tensor for illustration purpose.
    # In your own code, you can use the actual tensor from NN predicted output.

    # Inception v3 model
    inception_v3_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    inception_v3_pred[0, 0] = 0.261
    inception_v3_pred[0, 1] = 0.315
    inception_v3_pred[0, 2] = 0.102
    inception_v3_pred[0, 3] = 0.286

    # Xception model
    xception_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    xception_pred[0, 0] = 0.402
    xception_pred[0, 1] = 0.347
    xception_pred[0, 2] = 0.201
    xception_pred[0, 3] = 0.050

    # DenseNet-169
    densenet_169_pred = torch.zeros([1, 4])  # [Batch, Class Index]
    densenet_169_pred[0, 0] = 0.357
    densenet_169_pred[0, 1] = 0.467
    densenet_169_pred[0, 2] = 0.131
    densenet_169_pred[0, 3] = 0.045

    # Ensemble itself
    result = fuzzy_rank_ensemble(
        [inception_v3_pred, xception_pred, densenet_169_pred])
    print(f"Result: {result}")

```

### Monai

Note: in monai compatible wrapper, the Ensemble has been modified to be compatible with  `AsDiscrete(argmax=True)` function.

This is a simple example of how to use fuzzy rank ensemble with Monai.

```python
    import torch
    from fuzzy_rank_ensemble_torch.monai import FuzzyRankBasedEnsemble
    # The similar initialization as in the previous example.
    # ...

    ens = FuzzyRankBasedEnsemble()
    result = ens([inception_v3_pred, xception_pred, densenet_169_pred])
    assert torch.allclose(result, torch.tensor(
        [[1-0.4587, 1-0.4264, 1-0.5948, 1-0.5883],]), rtol=1e-03)
```

Also, you can use dictionary-based wrapper `FuzzyRankBasedEnsembled`.

## Citation

If this repository helps you in any way, consider citing our papers as follows:

```bib
@article{10.1016/j.bspc.2024.107342,
  title     = {A fuzzy rank-based ensemble of CNN models for MRI segmentation},
  journal   = {Biomedical Signal Processing and Control},
  volume    = {102},
  pages     = {107342},
  year      = {2025},
  issn      = {1746-8094},
  doi       = {10.1016/j.bspc.2024.107342},
  url       = {https://www.sciencedirect.com/science/article/pii/S1746809424014009},
  author    = {Daria Valenkova and Asya Lyanova and Aleksandr Sinitca and Ram Sarkar and Dmitrii Kaplun},
}
```

```bib
@article{manna2021fuzzy,
  title={A fuzzy rank-based ensemble of CNN models for classification of cervical cytology},
  author={Manna, Ankur and Kundu, Rohit and Kaplun, Dmitrii and Sinitca, Aleksandr and Sarkar, Ram},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={1--18},
  year={2021},
  doi={10.1038/s41598-021-93783-8},
  publisher={Nature Publishing Group}
}
```
