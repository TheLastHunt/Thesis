# main.py

"""
Entry point for Bokeh Server: assembles data, models, plots, and widgets.
"""

import os
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import CustomJS
from bokeh.events import ButtonClick

from data_loader import load_data, scale_data
from som_model import train_som, compute_umatrix
from cluster_analysis import assign_clusters, compute_cluster_means
from widgets import create_um_toggle, create_cluster_buttons
from plots import build_hex_plot, build_map_plot, build_data_table


# 1) Load & preprocess
HERE = os.path.dirname(__file__)
gpkg_path = os.path.join(HERE, 'data', 'mydata.gpkg')
gdf = load_data(gpkg_path)
scaled_df, geo_df = scale_data(gdf)

# 2) Train SOM & compute U-Matrix
som     = train_som(scaled_df)
um_flat = compute_umatrix(som)

# 3) Assign clusters & compute summary
hex_df           = assign_clusters(som, scaled_df)
cluster_means_df = compute_cluster_means(hex_df)

# 4) Create widgets
toggle          = create_um_toggle()
cluster_buttons = create_cluster_buttons(n_clusters=cluster_means_df.shape[0])

# 5) Build plots & table
# build_hex_plot now returns a ColumnDataSource of one row per SOM unit
p_hex, source_hex = build_hex_plot(hex_df, som, um_flat, toggle)

# pass BMU coords into geo_df so map_source has them for region selection
geo_with_bmu = geo_df.assign(bmu_x=hex_df['bmu_x'], bmu_y=hex_df['bmu_y'])
p_map, source_map = build_map_plot(geo_with_bmu, hex_df, cluster_buttons)

data_table   = build_data_table(cluster_means_df)
source_table = data_table.source

# 6a) Cluster‐button callbacks: select/deselect all units & regions in the cluster
for i, btn in enumerate(cluster_buttons):
    btn.js_on_event(ButtonClick, CustomJS(args=dict(
        hex_src=source_hex,
        map_src=source_map,
        table_src=source_table,
        cl=i
    ), code="""
        // gather unit indices for cluster cl
        const hex_inds = [];
        const hx = hex_src.data['hc_cluster'];
        for (let j = 0; j < hx.length; j++) {
            if (hx[j] === cl) hex_inds.push(j);
        }
        // gather region indices for cluster cl
        const map_inds = [];
        const mx = map_src.data['hc_cluster'];
        for (let j = 0; j < mx.length; j++) {
            if (mx[j] === cl) map_inds.push(j);
        }
        // toggle-off if already selected
        const already = table_src.selected.indices.length===1
                       && table_src.selected.indices[0]===cl;
        if (already) {
            hex_src.selected.indices   = [];
            map_src.selected.indices   = [];
            table_src.selected.indices = [];
        } else {
            hex_src.selected.indices   = hex_inds;
            map_src.selected.indices   = map_inds;
            table_src.selected.indices = [cl];
        }
        hex_src.change.emit();
        map_src.change.emit();
        table_src.change.emit();
    """))

# 6b) Hex‐plot → Map & Table connectivity: single‐unit selection
source_hex.selected.js_on_change('indices', CustomJS(args=dict(
    hex_src=source_hex,
    map_src=source_map,
    table_src=source_table
), code="""
    const inds = hex_src.selected.indices;
    if (hex_src.data._skip){
        hex_src.data._skip = false;
        return;
    }
    if (inds.length === 1 && hex_src.data._last_sel === inds[0]) {
        hex_src.data._skip = true;
        hex_src.selected.indices   = [];
        map_src.selected.indices   = [];
        table_src.selected.indices = [];
        hex_src.data._last_sel = null;
        hex_src.change.emit();
        map_src.change.emit();
        table_src.change.emit();
        return;
    }
    hex_src.data._last_sel = inds.length === 1 ? inds[0] : null;
    if (inds.length === 0) {
        map_src.selected.indices   = [];
        table_src.selected.indices = [];
    } else if (inds.length === 1) {
        const idx = inds[0];
        // find all regions with matching BMU coords
        const bx = hex_src.data['bmu_x'][idx];
        const by = hex_src.data['bmu_y'][idx];
        const xs = map_src.data['bmu_x'];
        const ys = map_src.data['bmu_y'];
        const region_inds = [];
        for (let i = 0; i < xs.length; i++) {
            if (xs[i] === bx && ys[i] === by) {
                region_inds.push(i);
            }
        }
        map_src.selected.indices   = region_inds;
        const cl = hex_src.data['hc_cluster'][idx];
        table_src.selected.indices = [cl];
    } else {
        return;
    }
    map_src.change.emit();
    table_src.change.emit();
"""))

# 6c) Map‐plot → Hex & Table connectivity: single‐unit selection
source_map.selected.js_on_change('indices', CustomJS(args=dict(
    hex_src=source_hex,
    map_src=source_map,
    table_src=source_table
), code="""
    const inds = map_src.selected.indices;
    if (map_src.data._skip){
        map_src.data._skip = false;
        return;
    }
    if (inds.length === 1 && map_src.data._last_sel === inds[0]) {
        map_src.data._skip = true;
        map_src.selected.indices   = [];
        hex_src.selected.indices   = [];
        table_src.selected.indices = [];
        map_src.data._last_sel = null;
        map_src.change.emit();
        hex_src.change.emit();
        table_src.change.emit();
        return;
    }
    map_src.data._last_sel = inds.length === 1 ? inds[0] : null;
    if (inds.length === 0) {
        hex_src.selected.indices   = [];
        table_src.selected.indices = [];
    } else if (inds.length === 1) {
        const idx = inds[0];
        const bx  = map_src.data['bmu_x'][idx];
        const by  = map_src.data['bmu_y'][idx];
        const xs  = hex_src.data['bmu_x'];
        const ys  = hex_src.data['bmu_y'];
        let hex_i = null;
        for (let i = 0; i < xs.length; i++) {
            if (xs[i] === bx && ys[i] === by) { hex_i = i; break; }
        }
        hex_src.selected.indices   = hex_i !== null ? [hex_i] : [];
        const cl = map_src.data['hc_cluster'][idx];
        table_src.selected.indices = [cl];
    } else {
        return;
    }
    hex_src.change.emit();
    table_src.change.emit();
"""))


# 7) Assemble layout
layout = column(
    row(toggle, *cluster_buttons, sizing_mode="stretch_width"),
    row(p_hex,    p_map,           sizing_mode="stretch_width"),
    data_table,
    sizing_mode="stretch_width"
)

curdoc().add_root(layout)
curdoc().title = "SOM Dashboard"
