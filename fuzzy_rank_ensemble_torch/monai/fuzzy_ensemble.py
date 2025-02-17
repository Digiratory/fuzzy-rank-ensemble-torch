from collections.abc import Sequence

from monai.config.type_definitions import (KeysCollection, NdarrayOrTensor)
from monai.transforms.post.array import Ensemble
from monai.transforms.post.dictionary import Ensembled
from monai.transforms.transform import Transform
from monai.utils import TransformBackends

from fuzzy_rank_ensemble_torch import fuzzy_rank_ensemble

__all__ = [
    "FuzzyRankBasedEnsemble",
    "FuzzyRankBasedEnsembled",
]


class FuzzyRankBasedEnsemble(Ensemble, Transform):
    """ Prediction are inverted (1 - FS) to be compatible with `AsDiscrete(argmax=True)`

    """
    backend = [TransformBackends.TORCH]

    def __init__(self) -> None:
        pass

    def __call__(self, img: Sequence[NdarrayOrTensor]) -> NdarrayOrTensor:
        out_pt = 1 - fuzzy_rank_ensemble(img)
        return self.post_convert(out_pt, img)


class FuzzyRankBasedEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    backend = FuzzyRankBasedEnsemble.backend

    def __init__(
            self,
            keys: KeysCollection,
            output_key: str | None = None) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
        """
        ensemble = FuzzyRankBasedEnsemble()
        super().__init__(keys, ensemble, output_key)
