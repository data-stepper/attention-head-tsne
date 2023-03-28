import functools
import warnings

import numpy as np
from typing import Optional, Union

from scipy.cluster._optimal_leaf_ordering import squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch.nn import Module
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertAttention,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download_model_from_huggingface(
    model_name: str = "bert-base-uncased",
) -> tuple[AutoTokenizer, Module]:
    """Download a model and a tokenizer from huggingface.

    Parameters
    ----------
    model_name : str
        The name of the model to download.

    Returns
    -------
    tuple[AutoTokenizer, AutoModel]
        A tuple of the tokenizer and model.

    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, from_tf=True)
    # experiments_model = AutoModel.from_pretrained(model_name, from_tf=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, from_tf=True
    )

    return tokenizer, model


def _tokens_from_text(
    text: str, tokenizer: AutoTokenizer, max_tokens: int = 512
) -> np.ndarray:
    return (
        tokenizer(text, return_tensors="pt").input_ids[0].numpy()[:max_tokens]
    )


def _fit_classical_tsne(
    model: torch.nn.Module,
    tokens: np.ndarray,
    perplexity: int = 20,
    n_iter: int = 1000,
    metric: str = "precomputed",
    method: str = "exact",
    layer: int = 11,
) -> TSNE:
    """Fit a classical t-SNE to the given model and tokens."""

    if not isinstance(model, BertForSequenceClassification):
        warnings.warn(
            "The given model is not an instance of BertForSequenceClassification. "
            "We cannot guarantee that the model is compatible with the "
            "current implementation of the AH-tSNE."
        )

    # Embed the tokens
    h = __embeddings_from_tokens(tokens, model, layer=layer)

    # Compute the distance matrix
    distance_matrix = np.linalg.norm(h[None, ...] - h[:, None, :], axis=-1)

    # Check that we use at maximum a perplexity of the number of tokens
    perp_: float = min(perplexity, len(tokens) - 1)

    # Fit the t-SNE
    t = TSNE(
        perplexity=perp_,
        n_iter=n_iter,
        metric=metric,
        method=method,
        init="random",
    ).fit(distance_matrix)

    return t


def _symmetrize_p_dist_matrix(attention_matrix: np.ndarray) -> np.ndarray:
    assert np.all(
        attention_matrix >= 0.0
    ), "Attention matrix must be non-negative."

    joint_distributions = (attention_matrix.T + attention_matrix) * (
        1.0 - np.eye(attention_matrix.shape[0])
    )
    joint_distributions = joint_distributions / joint_distributions.sum()

    return joint_distributions


def __embeddings_from_tokens(
    tokens: Union[list, np.ndarray],
    model: BertForSequenceClassification,
    layer: int = 10,
) -> np.ndarray:
    """Get the embeddings from the given tokens."""

    y = model.forward(
        input_ids=torch.tensor(tokens).unsqueeze(0), output_hidden_states=True
    )
    h = y.hidden_states[layer][0].detach().numpy()

    return h


def __recursively_get_children_modules_of_type(
    parent_module: Module, desired_class: type
) -> list:
    """Get all children modules of a parent module of a certain type.

    Parameters
    ----------
    parent_module : Module
        The parent module to get the children modules from.
    desired_class : type
        The type of the children modules to get.

    Returns
    -------
    list
        A list of all children modules of the desired type.

    """

    if not hasattr(parent_module, "children"):
        raise TypeError(
            "parent_module must be a Module or have a children attribute, not compatible"
            " with object of type: {}".format(type(parent_module))
        )

    children = list(parent_module.children())

    children_of_desired_type = []

    for child in children:
        if isinstance(child, desired_class):
            children_of_desired_type.append(child)

        else:
            children_of_desired_type += (
                __recursively_get_children_modules_of_type(child, desired_class)
            )

    return children_of_desired_type


def get_queries_and_keys_from_tokens(
    tokens: np.ndarray,
    model: BertForSequenceClassification,
    head: int = 2,
    layer: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the queries and keys from the given tokens.

    Parameters
    ----------
    tokens : np.ndarray
        The tokens to get the queries and keys from.
    model : BertForSequenceClassification
        The model to get the queries and keys from.
    head : int, optional
        The head of the attention mechanism to get the queries and keys from, by default 2

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of the queries and keys.

    """

    # Get the attention mechanism
    attention_mechanisms: list = __recursively_get_children_modules_of_type(
        model.bert.base_model, BertAttention
    )

    # Get the query and key matrices
    WQ: np.ndarray = (
        attention_mechanisms[layer].self.query.weight.detach().numpy()
    )
    WK: np.ndarray = (
        attention_mechanisms[layer].self.key.weight.detach().numpy()
    )

    # Embed the tokens
    h = __embeddings_from_tokens(tokens, model, layer=layer)

    # Compute the queries and keys
    queries = np.dot(h, WQ.T)[..., head * 64 : (head + 1) * 64]
    keys = np.dot(h, WK.T)[..., head * 64 : (head + 1) * 64]

    return queries, keys


def get_custom_joint_probabilities(
    tokens: np.ndarray,
    model: BertForSequenceClassification,
    head: int = 2,
    symmetrize: bool = True,
):
    queries, keys = get_queries_and_keys_from_tokens(tokens, model, head=head)

    attention_matrix = np.exp(np.dot(queries, keys.T))
    attention_matrix = attention_matrix / attention_matrix.sum(
        axis=1, keepdims=True
    )

    if not symmetrize:
        return attention_matrix

    return _symmetrize_p_dist_matrix(attention_matrix)


def fit_attention_head_based_tsne(
    tokens: np.ndarray,
    model: BertForSequenceClassification,
    layer: int = 2,
    head: int = 9,
    n_iter: int = 1000,
    metric: str = "precomputed",
    method: str = "exact",
    skip_num_points: int = 0,
    neighbors_nn: int = None,
    degrees_of_freedom: int = 1,
    fitted_tsne: Optional[TSNE] = None,
) -> TSNE:
    n_samples: int = tokens.shape[0]

    # We first need a fitted TSNE object to call the _tsne method
    if fitted_tsne is None:
        fitted_tsne: TSNE = _fit_classical_tsne(
            tokens=tokens,
            model=model,
            n_iter=n_iter,
            metric=metric,
            method=method,
        )

    # Create the matrix of joint probabilities
    # P: np.ndarray = squareform(
    #     _symmetrize_p_dist_matrix(get_attention_weights(*tokens)[layer, head])
    # )

    P: np.ndarray = squareform(
        get_custom_joint_probabilities(tokens, model, head=head)
    )

    # This is how we init the y_i in the beginning
    X_embedded: np.ndarray = 1e-4 * np.random.standard_normal(
        size=(n_samples, 2)
    )

    # Call it like this
    output: np.ndarray = fitted_tsne._tsne(
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded=X_embedded,
        skip_num_points=skip_num_points,
        neighbors=neighbors_nn,
    )

    fitted_tsne.embedding_ = output.copy()

    return fitted_tsne
