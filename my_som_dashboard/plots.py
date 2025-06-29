# plots.py

"""
Build Bokeh figures: hex plot, geographic map, and data table.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from minisom import MiniSom
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, GeoJSONDataSource,
    HoverTool, LinearColorMapper, ColorBar,
    FixedTicker, CustomJS, TableColumn, DataTable
)
from bokeh.palettes import Viridis256, Category10
from bokeh.events import ButtonClick
from typing import Tuple, List


def build_hex_plot(
    hex_df: pd.DataFrame,
    som: MiniSom,
    um_flat: np.ndarray,
    toggle
) -> Tuple[figure, ColumnDataSource]:
    from bokeh.models.glyphs import HexTile

    # 1) cluster palette
    n_clusters = int(hex_df['hc_cluster'].max()) + 1
    base = Category10[10]
    cluster_palette = (base * ((n_clusters // 10) + 1))[:n_clusters]
    cmap_hc = LinearColorMapper(palette=cluster_palette, low=0, high=n_clusters - 1)

    # 2) Build a DataFrame of one row per SOM‐unit (node)
    #    with its (i,j), hc_cluster, u_color, and color
    w = som.get_weights()             # shape (X, Y, features)
    X, Y, _ = w.shape
    # flatten node_labels from our clustering of weights
    flat_weights = w.reshape(X*Y, -1)
    # we need the same hierarchical clustering as in cluster_analysis
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=n_clusters).fit(flat_weights)
    node_labels = hc.labels_         # length X*Y

    records = []
    um_min, um_max = um_flat.min(), um_flat.max()
    for i in range(X):
        for j in range(Y):
            idx = i*Y + j
            d = um_flat[idx]
            norm = int((d - um_min)/(um_max - um_min)*255)
            records.append({
                'bmu_x':      i,
                'bmu_y':      j,
                'hc_cluster': int(node_labels[idx]),
                'u_color':    Viridis256[norm],
                'color':      cluster_palette[int(node_labels[idx])],
                'alpha':      1.0,
            })
    node_df = pd.DataFrame.from_records(records)
    node_df["display_color"] = node_df["color"]
    node_source = ColumnDataSource(node_df)

    # 3) build the figure
    p_hex = figure(
        title="SOM Units (Hexplot)",
        tools="pan,wheel_zoom,reset,tap",
        match_aspect=True, width=450, height=450,
        background_fill_color="#ffffff", outline_line_color="#cccccc"
    )
    p_hex.grid.visible = False
    p_hex.axis.visible = False

    # 4) one glyph per node now
    hex_renderer = p_hex.hex_tile(
        q="bmu_x", r="bmu_y", size=1, orientation="flattop",
        source=node_source,
        fill_color={"field": "display_color"},
        fill_alpha="alpha",
        line_color="#ffffff",
        line_width=0.5,
    )
    hex_renderer.hover_glyph = HexTile(
        q="bmu_x", r="bmu_y", size=1, orientation="flattop",
        fill_color={"field": "display_color"}, line_color="#000000"
    )
    hex_renderer.selection_glyph = HexTile(
        q="bmu_x", r="bmu_y", size=1, orientation="flattop",
        fill_color={"field": "display_color"}, line_color="#000000"
    )
    hex_renderer.nonselection_glyph = HexTile(
        q="bmu_x", r="bmu_y", size=1, orientation="flattop",
        fill_color={"field": "display_color"}, fill_alpha=0.2, line_color="#ffffff"
    )

    # 5) exactly one HoverTool, one tooltip
    p_hex.tools = [t for t in p_hex.tools if not isinstance(t, HoverTool)]
    hover = HoverTool(
        renderers=[hex_renderer],
        tooltips=[
            ("Unit",      "(@bmu_x, @bmu_y)"),
            ("HC Cluster","@hc_cluster"),
            ("U-Dist",    "@u_color"),
        ],
        point_policy="follow_mouse"
    )
    p_hex.add_tools(hover)

    # 6) color bars
    cb_cl = ColorBar(
        color_mapper=cmap_hc,
        ticker=FixedTicker(ticks=list(range(n_clusters))),
        major_label_overrides={i: str(i) for i in range(n_clusters)},
        label_standoff=12, border_line_color=None, location=(0,0)
    )
    cmap_u = LinearColorMapper(palette=Viridis256, low=um_min, high=um_max)
    cb_u = ColorBar(color_mapper=cmap_u, label_standoff=12,
                    location=(0,0), title="U-Matrix", visible=False)
    p_hex.add_layout(cb_cl, 'right')
    p_hex.add_layout(cb_u,  'right')

    # 7) wire up U-matrix toggle
    toggle.js_on_change('active', CustomJS(
        args=dict(src=node_source, bar_c=cb_cl, bar_u=cb_u),
        code="""
            const showU = cb_obj.active;
            const colors = showU ? src.data['u_color'] : src.data['color'];
            // assign a fresh array to ensure change detection
            src.data['display_color'] = colors.slice();
            bar_c.visible = !showU;
            bar_u.visible =  showU;
            src.change.emit();
        """
    ))

    return p_hex, node_source


def build_map_plot(
    geo_df: gpd.GeoDataFrame,
    hex_df: pd.DataFrame,
    cluster_buttons: List
) -> Tuple[figure, GeoJSONDataSource]:
    from bokeh.models import GeoJSONDataSource

    # attach cluster label by index
    df = geo_df.copy()
    df['hc_cluster'] = hex_df['hc_cluster'].values

    # GeoJSON source
    source_map = GeoJSONDataSource(geojson=df.to_json())

    # match the hex‐plot palette exactly
    max_c = int(df['hc_cluster'].max())
    base = Category10[10]
    palette = (base * ((max_c // 10) + 1))[:max_c+1]
    cmap    = LinearColorMapper(palette=palette, low=0, high=max_c)

    p_map = figure(
        title="Geographic Map (HC Clusters)",
        tools="pan,wheel_zoom,reset,tap",
        width=450, height=450, background_fill_color="#efefef"
    )
    p_map.axis.visible = False
    p_map.grid.visible = False

    patches = p_map.patches(
        'xs', 'ys', source=source_map,
        fill_color={'field': 'hc_cluster', 'transform': cmap},
        line_color="white", line_width=0.5, hover_line_color="black"
    )
    p_map.add_tools(HoverTool(renderers=[patches], tooltips=[("Cluster","@hc_cluster")]))

    return p_map, source_map


def build_data_table(cluster_means_df: pd.DataFrame) -> DataTable:
    source = ColumnDataSource(cluster_means_df)
    cols   = [TableColumn(field='hc_cluster', title='Cluster', width=60)]
    for c in cluster_means_df.columns.drop('hc_cluster'):
        cols.append(TableColumn(field=c, title=c, width=120))
    return DataTable(source=source, columns=cols, width=800, height=280,
                     fit_columns=False, index_position=None)
