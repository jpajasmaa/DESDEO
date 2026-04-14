import plotly.graph_objects as go
import plotly.express as ex
import plotly.io as pio
import polars as pl
import numpy as np
import plotly.colors as pcolors

# Set default renderer
# pio.renderers.default = "browser"

def visualize_pcp_clusters(options, points_arr, centers_arr, labels, n_predetermined, iter_n, current_mps=None):
    """
    Visualizes the multi-objective space using a Custom Parallel Coordinates Plot (PCP).
    Each axis is independently scaled between its Ideal and Nadir values.
    """
    fig = go.Figure()
    obj_names = list(options.fake_ideal.keys())
    x_vals = obj_names
    cluster_colors = pcolors.qualitative.Plotly  # Standard discrete colors

    # ---------------------------------------------------------
    # 1. INDEPENDENT AXIS SCALING (Normalization)
    # ---------------------------------------------------------
    ideal = options.fake_ideal
    nadir = options.fake_nadir

    # Define the absolute bottom and top for each axis
    mins = {k: min(ideal[k], nadir[k]) for k in obj_names}
    maxs = {k: max(ideal[k], nadir[k]) for k in obj_names}
    ranges = {k: (maxs[k] - mins[k]) if (maxs[k] - mins[k]) != 0 else 1e-9 for k in obj_names}

    # Translates real numbers into a 0-to-1 scale for the shared plot
    def norm_point(p_array):
        return [(p_array[i] - mins[name]) / ranges[name] for i, name in enumerate(obj_names)]

    # Creates a hover tooltip showing the REAL unscaled numbers
    def make_hover(p_array, label_prefix):
        return "<br>".join([f"{name}: {val:.3f}" for name, val in zip(obj_names, p_array)])

    # ---------------------------------------------------------
    # 2. DRAW BACKGROUND AXES & REAL-VALUE LABELS
    # ---------------------------------------------------------
    for name in obj_names:
        # Draw the vertical axis line
        fig.add_trace(go.Scatter(
            x=[name, name], y=[0, 1], mode='lines',
            line=dict(color='lightgrey', width=2), showlegend=False, hoverinfo='skip'
        ))
        # Add the Real Numbers to the top and bottom of each independent axis
        fig.add_annotation(x=name, y=1.05, text=f"{maxs[name]:.2f}", showarrow=False, font=dict(size=12, color='black'))
        fig.add_annotation(x=name, y=-0.05, text=f"{mins[name]:.2f}", showarrow=False, font=dict(size=12, color='black'))

    # ---------------------------------------------------------
    # 3. EVALUATED POINTS (Faint lines colored by Cluster ID)
    # ---------------------------------------------------------
    added_clusters_to_legend = set()

    for i, p in enumerate(points_arr):
        cluster_idx = labels[i]
        c_color = cluster_colors[cluster_idx % len(cluster_colors)]
        normed_p = norm_point(p)  # Scale to independent axis

        show_in_legend = cluster_idx not in added_clusters_to_legend
        if show_in_legend:
            added_clusters_to_legend.add(cluster_idx)

        fig.add_trace(go.Scatter(
            x=x_vals, y=normed_p, mode='lines',
            line=dict(color=c_color, width=1.5),
            opacity=0.20,
            text=[make_hover(p, "Evaluated")] * len(x_vals),
            hoverinfo='text',
            name=f'Cluster {cluster_idx} Points',
            legendgroup=f'points_{cluster_idx}',  # Toggle all lines at once
            showlegend=show_in_legend
        ))

    # ---------------------------------------------------------
    # 4. FAIR SOLUTIONS (Pre-centers) - Matches Cluster Color
    # ---------------------------------------------------------
    pre_centers = centers_arr[:n_predetermined]
    for i, p in enumerate(pre_centers):
        c_color = cluster_colors[i % len(cluster_colors)]
        normed_p = norm_point(p)

        fig.add_trace(go.Scatter(
            x=x_vals, y=normed_p, mode='lines+markers',
            line=dict(color=c_color, width=4),
            marker=dict(symbol='square', size=10, color=c_color, line=dict(color='black', width=2)),
            name=f'Pre{i} (Fair Center)',
            text=[make_hover(p, f"Pre{i}")] * len(x_vals),
            hoverinfo='text+name',
            opacity=1.0
        ))

    # ---------------------------------------------------------
    # 5. NEW CANDIDATES - Dashed Blue Lines
    # ---------------------------------------------------------
    new_centers = centers_arr[n_predetermined:]
    for i, p in enumerate(new_centers):
        normed_p = norm_point(p)
        fig.add_trace(go.Scatter(
            x=x_vals, y=normed_p, mode='lines+markers',
            line=dict(color='blue', width=3, dash='dash'),
            marker=dict(symbol='diamond', size=10, color='blue', line=dict(color='white', width=2)),
            name=f'New{i} (Candidate)',
            text=[make_hover(p, f"New{i}")] * len(x_vals),
            hoverinfo='text+name',
            opacity=1.0
        ))

    # ---------------------------------------------------------
    # 6. FAKE IDEAL / NADIR BOUNDARIES
    # ---------------------------------------------------------
    fake_ideal_arr = [options.fake_ideal[k] for k in obj_names]
    fake_nadir_arr = [options.fake_nadir[k] for k in obj_names]

    fig.add_trace(go.Scatter(
        x=x_vals, y=norm_point(fake_ideal_arr), mode='lines+markers',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(symbol='triangle-down', size=12),
        name='fake_ideal',
        text=[make_hover(fake_ideal_arr, "Ideal")] * len(x_vals),
        hoverinfo='text+name'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=norm_point(fake_nadir_arr), mode='lines+markers',
        line=dict(color='orange', width=2, dash='dot'),
        marker=dict(symbol='triangle-up', size=12),
        name='fake_nadir',
        text=[make_hover(fake_nadir_arr, "Nadir")] * len(x_vals),
        hoverinfo='text+name'
    ))

    # ---------------------------------------------------------
    # 7. CURRENT DM ANCHORS (Distinct Colors)
    # ---------------------------------------------------------
    if current_mps is not None:
        dm_colors = pcolors.qualitative.Vivid

        for idx, (dm_name, obj_vals) in enumerate(current_mps.items()):
            mps_arr = [obj_vals[k] for k in obj_names]
            normed_mps = norm_point(mps_arr)
            dm_color = dm_colors[idx % len(dm_colors)]

            fig.add_trace(go.Scatter(
                x=x_vals, y=normed_mps, mode='markers',
                marker=dict(symbol='x', size=16, color=dm_color, line=dict(width=3, color='black')),
                name=f'{dm_name} Preferred',
                text=[make_hover(mps_arr, dm_name)] * len(x_vals),
                hoverinfo='text+name'
            ))

    # ---------------------------------------------------------
    # 8. LAYOUT FORMATTING
    # ---------------------------------------------------------
    fig.update_layout(
        title=f"Iteration {iter_n})",
        xaxis=dict(title="Objectives", showgrid=False),
        yaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            showticklabels=False,  # Hide the fake 0-to-1 scale!
            range=[-0.15, 1.15]   # Pad the top and bottom so our text annotations fit perfectly
        ),
        width=1200, height=1000,
        hovermode="closest",
        template="plotly_white"
    )

    return fig

def visualize_pcp_clusters_old(options, points_arr, centers_arr, labels, n_predetermined, iter_n, current_mps=None):
    """
    Visualizes the multi-objective space using a Custom Parallel Coordinates Plot (PCP).
    Supports any number of dimensions (e.g., the 5-objective River Pollution Problem).
    """
    fig = go.Figure()
    obj_names = list(options.fake_ideal.keys())
    x_vals = obj_names

    cluster_colors = pcolors.qualitative.Plotly  # Standard discrete colors

    # 1. Evaluated Points (Faint lines colored by Cluster ID & Grouped in Legend)
    added_clusters_to_legend = set()

    for i, p in enumerate(points_arr):
        cluster_idx = labels[i]
        c_color = cluster_colors[cluster_idx % len(cluster_colors)]

        # Only show the legend entry for the first line of each cluster
        show_in_legend = cluster_idx not in added_clusters_to_legend
        if show_in_legend:
            added_clusters_to_legend.add(cluster_idx)

        fig.add_trace(go.Scatter(
            x=x_vals, y=p, mode='lines',
            line=dict(color=c_color, width=1.5),
            opacity=0.25,
            hoverinfo='skip',  # Speeds up performance for 1000s of lines
            name=f'Cluster {cluster_idx} Points',
            legendgroup=f'points_{cluster_idx}',  # Links all lines in this cluster to one toggle!
            showlegend=show_in_legend
        ))

    # 2. Fair Solutions (Pre-centers) - Colored to match their cluster!
    pre_centers = centers_arr[:n_predetermined]
    for i, p in enumerate(pre_centers):
        c_color = cluster_colors[i % len(cluster_colors)]  # Match exact color of the point cloud

        fig.add_trace(go.Scatter(
            x=x_vals, y=p, mode='lines+markers',
            line=dict(color=c_color, width=4),
            marker=dict(symbol='square', size=10, color=c_color, line=dict(color='black', width=2)),
            name=f'Candidate {i}',
            opacity=1.0
        ))

    # 3. New Candidates - Dashed Blue Lines with Diamonds
    new_centers = centers_arr[n_predetermined:]
    for i, p in enumerate(new_centers):
        fig.add_trace(go.Scatter(
            x=x_vals, y=p, mode='lines+markers',
            line=dict(color='blue', width=3, dash='dash'),
            marker=dict(symbol='diamond', size=10, color='blue', line=dict(color='white', width=2)),
            name=f'New{i} (Candidate)',
            opacity=1.0
        ))

    # 4. Fake Ideal / Nadir Boundaries
    fake_ideal_arr = [options.fake_ideal[k] for k in obj_names]
    fake_nadir_arr = [options.fake_nadir[k] for k in obj_names]

    fig.add_trace(go.Scatter(
        x=x_vals, y=fake_ideal_arr, mode='lines+markers',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(symbol='triangle-down', size=12),
        name='ideal'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=fake_nadir_arr, mode='lines+markers',
        line=dict(color='orange', width=2, dash='dot'),
        marker=dict(symbol='triangle-up', size=12),
        name='nadir'
    ))

    # 5. Current DM Anchors (Magenta Crosses)
    if current_mps is not None:
        # Use a secondary, vibrant palette so DMs don't visually blend into the clusters
        dm_colors = pcolors.qualitative.Vivid
        for idx, (dm_name, obj_vals) in enumerate(current_mps.items()):
            mps_arr = [obj_vals[k] for k in obj_names]
            dm_color = dm_colors[idx % len(dm_colors)]  # Grab a unique color

            fig.add_trace(go.Scatter(
                x=x_vals, y=mps_arr, mode='markers',
                # Black outline added to the cross to make it "pop" on top of the lines
                marker=dict(symbol='x', size=14, color=dm_color, line=dict(width=3, color='black')),
                name=f'{dm_name} Preferred'
            ))

    fig.update_layout(
        title=f"Iteration {iter_n}",
        xaxis_title="Objectives",
        yaxis_title="Objective Values",
        width=1200, height=1000,
        hovermode="x unified",  # Tooltip draws a vertical line to compare points easily
        template="plotly_white"
    )

    return fig


def visualize_selection_2d(all_points, seed_solutions, new_solutions, title):
    """
    Plots the candidate pool and selected points in 2D.
    """
    def get_coords(point_list, is_evaluated_point=True):
        xs, ys = [], []
        for p in point_list:
            d = p.objectives if is_evaluated_point else p.objective_values
            xs.append(d["f_1"])
            ys.append(d["f_2"])
        return xs, ys

    fig = go.Figure()

    # 1. Plot All Candidates (Grey Background)
    cx, cy = get_coords(all_points, True)
    fig.add_trace(go.Scatter(
        x=cx, y=cy,
        mode='markers',
        name='Evaluated Points Space',
        marker=dict(size=6, color='lightgrey', opacity=0.6)
    ))

    # 2. Plot Seed Solution (Red Star)
    sx, sy = get_coords(seed_solutions, False)
    fig.add_trace(go.Scatter(
        x=sx, y=sy,
        mode='markers',
        name='Existing Fair Solution',
        marker=dict(size=12, color='red', symbol='star')
    ))

    # 3. Plot Newly Selected Points (Blue Circles with Numbers)
    nx, ny = get_coords(new_solutions, False)
    fig.add_trace(go.Scatter(
        x=nx, y=ny,
        mode='markers+text',
        name='Selected Candidates',
        text=[str(i+1) for i in range(len(new_solutions))],
        textposition="top center",
        textfont=dict(size=14, color="black"),
        marker=dict(size=10, color='blue', symbol='circle', line=dict(width=2, color='black'))
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Objective f_1',
        yaxis_title='Objective f_2',
        width=800, height=600,
        template="plotly_white"
    )
    fig.show(renderer="browser")


def visualize_3d_clusters(options, points_arr, centers_arr, labels, n_predetermined, iter_n, axis_limits=[0.0, 1.]):
    """
    Visualizes 3D clusters by coloring points.
    """
    fig = go.Figure()

    # 1. Plot Evaluated Points (Colored by Cluster)
    fig.add_trace(go.Scatter3d(
        x=points_arr[:, 0],
        y=points_arr[:, 1],
        z=points_arr[:, 2],
        mode='markers',
        name='Evaluated Points',
        marker=dict(
            size=4,
            color=labels,       # Color by cluster ID
            colorscale='Viridis',  # Distinct colors
            opacity=0.6
        ),
        text=[f"Cluster: {l}" for l in labels]  # Hover info
    ))

    # 2. Plot Predetermined Centers (Red Squares)
    pre_centers = centers_arr[:n_predetermined]
    fig.add_trace(go.Scatter3d(
        x=pre_centers[:, 0],
        y=pre_centers[:, 1],
        z=pre_centers[:, 2],
        mode='markers+text',
        name='Fair Solutions',
        text=[f"Pre{i}" for i in range(len(pre_centers))],
        textposition="top center",
        marker=dict(size=10, color='red', symbol='square', line=dict(width=2, color='black'))
    ))

    # 3. Plot New Candidates (Blue Diamonds)
    new_centers = centers_arr[n_predetermined:]
    fig.add_trace(go.Scatter3d(
        x=new_centers[:, 0],
        y=new_centers[:, 1],
        z=new_centers[:, 2],
        mode='markers+text',
        name='New Candidates',
        text=[f"New{i}" for i in range(len(new_centers))],
        textposition="top center",
        marker=dict(size=10, color='blue', symbol='diamond', line=dict(width=2, color='white'))
    ))

    # 4. Fake Ideal/Nadir
    fig.add_trace(go.Scatter3d(
        x=[options.fake_ideal["f_1"]], y=[options.fake_ideal["f_2"]], z=[options.fake_ideal["f_3"]],
        mode="markers", name="fake_ideal", marker_symbol="diamond", opacity=0.9
    ))

    fig.add_trace(go.Scatter3d(
        x=[options.fake_nadir["f_1"]], y=[options.fake_nadir["f_2"]], z=[options.fake_nadir["f_3"]],
        mode="markers", name="fake_nadir", marker_symbol="diamond", opacity=0.9
    ))

    fig.update_layout(
        title=f"Iteration {iter_n}: Clusters",
        scene=dict(
            xaxis=dict(title='f_1', range=axis_limits),
            yaxis=dict(title='f_2', range=axis_limits),
            zaxis=dict(title='f_3', range=axis_limits)
        ),
        width=1200, height=1000
    )
    # fig.show(renderer="browser")
    fig.show()


def visualize_3d(options, evaluated_points, fair_sols, n):
    fig = ex.scatter_3d()

    # Add reference points
    chosen_refps = pl.DataFrame([point.reference_point for point in evaluated_points])
    # rescale reference points
    chosen_refps = chosen_refps.with_columns(
        [
            (pl.col(obj) * (options.fake_nadir[obj] - options.fake_ideal[obj]) + options.fake_ideal[obj]).alias(obj)
            for obj in options.fake_ideal.keys()
        ]
    )

    fig = fig.add_scatter3d(
        x=chosen_refps["f_1"].to_numpy(),
        y=chosen_refps["f_2"].to_numpy(),
        z=chosen_refps["f_3"].to_numpy(),
        name="Reference Points",
        mode="markers",
        marker_symbol="circle",
        opacity=0.8,
    )
    # Add front
    front = pl.DataFrame([point.objectives for point in evaluated_points])
    fig = fig.add_scatter3d(
        x=front["f_1"].to_numpy(),
        y=front["f_2"].to_numpy(),
        z=front["f_3"].to_numpy(),
        mode="markers",
        name="Front",
        marker_symbol="circle",
        opacity=0.9,
    )
    fig = fig.add_scatter3d(
        x=[options.fake_ideal["f_1"]], y=[options.fake_ideal["f_2"]], z=[options.fake_ideal["f_3"]],
        mode="markers", name="fake_ideal", marker_symbol="diamond", opacity=0.9,
    )
    fig = fig.add_scatter3d(
        x=[options.fake_nadir["f_1"]], y=[options.fake_nadir["f_2"]], z=[options.fake_nadir["f_3"]],
        mode="markers", name="fake_nadir", marker_symbol="diamond", opacity=0.9,
    )
    DMs = options.most_preferred_solutions.keys()
    for dm in DMs:
        fig = fig.add_scatter3d(
            x=[options.most_preferred_solutions[dm]["f_1"]],
            y=[options.most_preferred_solutions[dm]["f_2"]],
            z=[options.most_preferred_solutions[dm]["f_3"]],
            mode="markers", name=dm, marker_symbol="square", opacity=0.9,
        )

    fair_crits = [fair_sols[i].fairness_criterion for i in range(len(fair_sols))]
    for i, fc in enumerate(fair_crits):
        fig = fig.add_scatter3d(
            x=[fair_sols[i].objective_values["f_1"]],
            y=[fair_sols[i].objective_values["f_2"]],
            z=[fair_sols[i].objective_values["f_3"]],
            mode="markers", name=fc, marker_symbol="x", opacity=0.9,
        )

    fig.layout.scene.camera.projection.type = "orthographic"
    fig.update_layout(autosize=False, width=1200, height=1200)
    fig.show(renderer="browser")


def visualize_expansion(points_matrix, winning_points, new_candidates, winning_center, winning_idx, fraction_to_keep, axis_limits=[-0.2, 2]):
    """Visualizes the Original Cluster points and the New Generated Candidates."""
    fig = go.Figure()

    # Plot All Points (faint background)
    fig.add_trace(go.Scatter3d(
        x=points_matrix[:, 0], y=points_matrix[:, 1], z=points_matrix[:, 2],
        mode='markers', name='All Evaluated Points',
        marker=dict(size=3, color='grey', opacity=0.4)
    ))

    # Plot Winning Cluster (Green)
    fig.add_trace(go.Scatter3d(
        x=winning_points[:, 0], y=winning_points[:, 1], z=winning_points[:, 2],
        mode='markers', name=f'Winning Cluster (Idx {winning_idx})',
        marker=dict(size=5, color='green', opacity=0.8)
    ))

    # Plot New Candidates (Red X)
    fig.add_trace(go.Scatter3d(
        x=new_candidates[:, 0], y=new_candidates[:, 1], z=new_candidates[:, 2],
        mode='markers', name='New Expanded Candidates',
        marker=dict(size=6, color='red', opacity=1.0)
    ))

    # Plot the Winning Center (for reference)
    fig.add_trace(go.Scatter3d(
        x=[winning_center[0]], y=[winning_center[1]], z=[winning_center[2]],
        mode='markers', name='Winning Center',
        marker=dict(size=10, color='gold', symbol='diamond')
    ))
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    # 5. Calculate Hulls for Edges
    try:
        hull_inner = ConvexHull(winning_points, qhull_options='QJ')
        hull_outer = ConvexHull(new_candidates, qhull_options='QJ')

        inner_verts = winning_points[hull_inner.vertices]
        outer_verts = new_candidates[hull_outer.vertices]

        line_x, line_y, line_z = [], [], []
        text_x, text_y, text_z = [], [], []
        text_labels = []

        # Helper to add a link
        def add_link(v_from, v_to):
            # Line Geometry
            line_x.extend([v_from[0], v_to[0], None])
            line_y.extend([v_from[1], v_to[1], None])
            line_z.extend([v_from[2], v_to[2], None])

            # Angle Calculation
            vec_c_from = v_from - winning_center
            vec_c_to = v_to - winning_center
            norm_from = np.linalg.norm(vec_c_from)
            norm_to = np.linalg.norm(vec_c_to)

            angle_deg = 0.0
            if norm_from > 1e-9 and norm_to > 1e-9:
                cos_theta = np.clip(np.dot(vec_c_from, vec_c_to) / (norm_from * norm_to), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

            midpoint = (v_from + v_to) / 2
            text_x.append(midpoint[0])
            text_y.append(midpoint[1])
            text_z.append(midpoint[2])
            text_labels.append(f"{angle_deg:.0f}°")

        # --- DIRECTION 1: Outer -> Nearest Inner ---
        dists_out_to_in = cdist(outer_verts, inner_verts)
        closest_inner_indices = np.argmin(dists_out_to_in, axis=1)

        for i, inner_idx in enumerate(closest_inner_indices):
            add_link(inner_verts[inner_idx], outer_verts[i])

        # --- DIRECTION 2: Inner -> Nearest Outer ---
        # We assume symmetry is desired: ensure every inner vertex also links to its closest outer neighbor
        dists_in_to_out = cdist(inner_verts, outer_verts)
        closest_outer_indices = np.argmin(dists_in_to_out, axis=1)

        for i, outer_idx in enumerate(closest_outer_indices):
            # Optional: Check if we already drew this line to avoid duplicates?
            # For visualization, drawing twice is harmless and simpler.
            add_link(inner_verts[i], outer_verts[outer_idx])        # Add Lines Trace

        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines',
            name='Expansion Links',
            line=dict(color='black', width=2, dash='dot')
        ))

        # Add Angle Labels Trace
        fig.add_trace(go.Scatter3d(
            x=text_x, y=text_y, z=text_z,
            mode='text',
            name='Angles',
            text=text_labels,
            textfont=dict(color='black', size=10)
        ))

    except Exception as e:
        print(f"Could not visualize hull edges: {e}")

    fig.update_layout(
        title=f"Convex Hull Expansion (Top {fraction_to_keep*100}%)",
        scene=dict(
            xaxis=dict(title='f_1', range=axis_limits),
            yaxis=dict(title='f_2', range=axis_limits),
            zaxis=dict(title='f_3', range=axis_limits)
        ),
        width=1200, height=1000
    )
    # fig.write_html("extended_cvh.html")
    # fig.show(renderer="browser")
    fig.show()
