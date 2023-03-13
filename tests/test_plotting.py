import pandas as pd
import unittest
from src.ah_tsne.plotting import better_scatter_plot


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [1, 2, 3, 4, 5],
                "label": ["a", "b", "c", "d", "e"],
                "hue": ["A", "B", "C", "D", "E"],
            }
        )

    def test_standard_plot(self):
        better_scatter_plot(
            df=self.dummy_df,
            label_col="label",
            x_col="x",
            y_col="y",
            hue="hue",
            plot_title="Test",
            ax=None,
            n_quantiles=50,
            n_max_ticks=30,
            apply_pca=False,
            ticks_as_percentiles=False,
            rescale_axis=True,
        )


if __name__ == "__main__":
    unittest.main()
