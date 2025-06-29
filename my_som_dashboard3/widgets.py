# widgets.py

"""
Define interactive widgets: U-Matrix toggle and cluster-selection buttons.
"""

from bokeh.models import Toggle, Button
from bokeh.palettes import Category10
from typing import List

def create_um_toggle() -> Toggle:
    """
    Toggle between cluster and U-Matrix coloring.
    """
    return Toggle(label="Show U-Matrix", button_type="primary", active=False)


def create_cluster_buttons(n_clusters: int) -> List[Button]:
    """
    Generate a list of Buttons for selecting clusters.

    Parameters:
        n_clusters: how many cluster buttons to create.

    Returns:
        A list of Bokeh Button widgets, each with a CSS class 'cluster-btn-{i}'.
    """
    base = Category10[10]
    palette = (base * ((n_clusters // 10) + 1))[:n_clusters]

    buttons: List[Button] = []
    for i in range(n_clusters):
        btn = Button(label=str(i), css_classes=[f"cluster-btn-{i}"], width=30)
        buttons.append(btn)
    return buttons
