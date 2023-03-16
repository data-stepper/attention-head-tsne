from src.ah_tsne.__main__ import main
import pytest

sample_text: str = """
Usually you want to be able to access these applications from anywhere on your system,
but installing packages and their dependencies to the same global environment can cause 
version conflicts and break dependencies the operating system has on Python packages.
"""

models_to_test: list[str] = [
    "bert-base-uncased",
    "bert-base-cased",
]

layers_to_test: list[int] = [0, 2, 11]
heads_to_test: list[int] = [0, 11]


@pytest.mark.parametrize("model_name", models_to_test)
@pytest.mark.parametrize("layer", layers_to_test)
@pytest.mark.parametrize("head", heads_to_test)
def test_main(model_name: str, layer: int, head: int) -> None:
    main(
        text=sample_text,
        model_name=model_name,
        layer=layer,
        head=head,
    )
