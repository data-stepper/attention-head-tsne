from unittest import TestCase
from sklearn.manifold import TSNE

import numpy as np

from ah_tsne.util import (
    _fit_classical_tsne,
    get_custom_joint_probabilities,
    fit_attention_head_based_tsne,
    download_model_from_huggingface,
    _tokens_from_text,
)


class Test(TestCase):
    def setUp(self) -> None:
        self.sample_text: str = "This is a sample text."
        self.model_name: str = "bert-base-uncased"

        self.tokenizer, self.model = download_model_from_huggingface(
            self.model_name
        )
        self.tokens: np.ndarray = _tokens_from_text(
            self.sample_text, self.tokenizer
        )

    def test_fit_classical_tsne(self):
        t = _fit_classical_tsne(
            model=self.model,
            tokens=self.tokens,
        )

        assert isinstance(t, TSNE)
        assert hasattr(t, "embedding_")
        assert isinstance(t.embedding_, np.ndarray)

    def test_get_custom_joint_probabilities(self):
        p = get_custom_joint_probabilities(
            tokens=self.tokens,
            model=self.model,
            head=2,
            symmetrize=True,
        )

        assert isinstance(p, np.ndarray)
        assert p.shape == (len(self.tokens), len(self.tokens))
        assert np.all(p >= 0)
        assert np.all(p <= 1)
        assert np.sum(p) == 1
        assert np.allclose(p, p.T)

    def test_fit_attention_head_based_tsne(self):
        t = fit_attention_head_based_tsne(
            tokens=self.tokens,
            model=self.model,
            layer=2,
            head=9,
            n_iter=1000,
            metric="precomputed",
            method="exact",
            skip_num_points=0,
            neighbors_nn=None,
            degrees_of_freedom=1,
            fitted_tsne=None,
        )

        assert isinstance(t, TSNE)
        assert hasattr(t, "embedding_")
        assert isinstance(t.embedding_, np.ndarray)
