#!/usr/bin/env python

import typer
from .util import (
    fit_attention_head_based_tsne,
    download_model_from_huggingface,
    __embeddings_from_tokens,
    _tokens_from_text,
)
from .plotting import better_scatter_plot


def main(
    text: str = typer.Argument(..., help="The text to visualize."),
    model_name: str = typer.Option(
        "bert-base-uncased",
        help="The name of the model to use.",
    ),
    layer: int = typer.Option(
        11,
        help="The layer whose embeddings to use.",
    ),
    head: int = typer.Option(
        9,
        help="The attention head on that layer to use.",
    ),
) -> None:
    """Plot the attention head based t-SNE visualization of a text.

    The Attention head based t-SNE algorithm can be used to visualize
    arbitrary (preferably english) texts which are shorter than the
    maximum sequence length of the used model (512 in the case of BERT).
    """
    # TODO: Write a nicer text here
    # TODO: Write a clipboard summarizer here, that's really useful and cool

    tokenizer, model = download_model_from_huggingface(model_name)
    max_seq_len: int = model.config.seq_len

    print("Only using the first {} tokens of the text.".format(max_seq_len))

    tokens = _tokens_from_text(text, tokenizer, max_tokens=max_seq_len)

    # TODO: Transform this again to the dataframe containing the embeddings
    # and the original token strings so we can plot them with the better_scatter_plot


if __name__ == "__main__":
    typer.run(main)
