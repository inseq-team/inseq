from typing import NoReturn

import matplotlib.pyplot as plt
import seaborn as sns

from amseq import GradientAttributionOutput


def heatmap(attr: GradientAttributionOutput, cmap=None, figsize=(12, 5)) -> NoReturn:
    if cmap is None:
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
    plt.subplots(figsize=figsize)
    sns.heatmap(
        attr.attributions,
        xticklabels=attr.target_tokens,
        yticklabels=attr.source_tokens,
        cmap=cmap,
    )
