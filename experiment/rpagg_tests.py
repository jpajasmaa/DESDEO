from desdeo.problem import Problem, objective_dict_to_numpy_array, numpy_array_to_objective_dict, get_ideal_dict, get_nadir_dict
from desdeo.tools.ng_solver_interfaces import NevergradGenericOptions
from desdeo.tools.scalarization import add_asf_diff, add_guess_sf_diff, add_stom_sf_diff, add_asf_generic_diff, add_asf_generic_nondiff, add_guess_sf_nondiff
import numpy as np

from desdeo.problem.testproblems import zdt1, zdt2, zdt3, dtlz2
# noqa

from desdeo.tools.scipy_solver_interfaces import ScipyMinimizeOptions
from desdeo.tools.utils import guess_best_solver, PyomoIpoptSolver, NevergradGenericSolver

from preference_aggregation import find_GRP, maxmin_cones_criterion, maxmin_criterion


import matplotlib.pyplot as plt

from pymoo.problems import get_problem
from pymoo.util.plotting import plot

import plotly.express as px

import pandas as pd

import plotly.io as pio
pio.kaleido.scope.mathjax = None


pdf_path = "/home/jp/tyot/mop/papers/prefagg_concept/oldversion/experiment_pics/paperpics/pdfs/"
html_path = "/home/jp/tyot/mop/papers/prefagg_concept/oldversion/experiment_pics/paperpics/htmls/"
csv_path = "/home/jp/tyot/mop/papers/prefagg_concept/oldversion/exptables/"

def visualize(problem_name, data, column_names, proj=False):
    # visualize
    marker_size = 15

    if proj:
        problem_pymoo = get_problem(problem_name)
        # problem_pymoo = get_problem(problem.name, n_var=30, n_obj=2)
        pf = problem_pymoo.pareto_front()

        keys = ["f_1", "f_2"]
        # Convert NumPy array to a list of dictionaries
        PF = [dict(zip(keys, row)) for row in pf]
        # plot PF
        if problem_name == "zdt3":
            fig = px.scatter(PF, x="f_1", y="f_2")
        else:
            fig = px.line(PF, x="f_1", y="f_2")

    else:
        df_empty = {
            "f_1": [data["cip"]["f_1"]],
            "f_2": [data["cip"]["f_2"]],
        }
        fig = px.scatter(df_empty, x="f_1", y="f_2")

    # data.index.name = 'Group'
    # data = data.reset_index().rename(columns={'index': 'Group'})

    import plotly.graph_objects as go
    # data = data.reset_index()

    print(data)

    # TODO: do smarter, there cna be more DMs
    # list_of_rows = ["ideal","cip","DM1", "DM2", "DM3", "meanGRP","addGRP","conesGRP"]
    colors = []
    markers = []

    for name in column_names:

        fig.add_trace(
            go.Scatter(
                x=[data.loc[name, "f_1"]],
                y=[data.loc[name, "f_2"]],
                mode='markers',
                name=name,
                # color='Group',  # This assigns a unique color to each group
                text=f'{name}',  # This displays the group name next to each point
                # marker=dict(size=10, color=name)
            )
        )
    """
    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        if proj:
            fig.add_scatter(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], mode="markers", name=f"proj. DM{
                            i+1}_RP", showlegend=True, marker=dict(size=10, symbol="circle"))
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=marker_size, symbol="circle"))

    # PLOT GRP
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="GRP-mean", showlegend=True, marker=dict(size=marker_size, symbol="square"))
    # additive
    # fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="GRP_mm_ext", showlegend=True, marker=dict(size=marker_size, symbol="x"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="GRP-add", showlegend=True, marker=dict(size=marker_size, symbol="x"))
    # cones
    # fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="GRP-cones_ext",
    #                showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="GRP-cones", showlegend=True, marker=dict(size=marker_size, symbol="cross"))

    # lines to show the polyhedron
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # FOR saving as pdf
    fig.update_layout(autosize=True, width=800, height=800, title=f"{name}")
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    """
    """
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    """  # for bottom right legend
    # fig.update_xaxes(range=[0.1, 0.8])
    # fig.update_yaxes(range=[0.1, 0.8])
    # fig.update_layout(margin_r=300, legend_x=1.2, autosize=True, width=800, height=800, title=f"{name}")
    # fig.write_image(f"{pdf_path}{name}.pdf", width=800, height=800)
    # fig.write_html(f"{html_path}{name}.html")

    fig.show()


def solve_test_setting(problem: Problem, rps: list[dict[str, float]], cip, name, proj=False, solver=None, solver_options=None):
    # nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    k = len(ideal)
    q = len(rps)

    # TODO: make for any number of objs to work
    # rp_arr = np.array([[col["f_1"], col["f_2"], col["f_3"], col["f_4"]] for col in rps])
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])

    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    print("CIP", cip)
    print(ideal)
    print(rps)

    GRP = None
    GRP_ext = None
    GRP_cones = None
    GRP_cones_ext = None

    if proj:
        converted_prefs = []
        # TODO: find PO for each DM RP, for now only for asf, plotting mess
        for rp in range(len(rp_arr)):
            dm_rp = cip - rp_arr[rp]  # convert to improvement direction
            # should convert numpy array (rp) of dm to dict.
            dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
            # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
            p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
                problem,
                symbol=f"asf{rp}",
                reference_point=cip_dict,
                weights=dm_rp,
                reference_point_aug=cip_dict,
            )
            # for 1 RP
            solver = PyomoIpoptSolver(p)
            res = solver.solve(target)
            fs = res.optimal_objectives
            # print(fs)
            converted_prefs.append(fs)

        print(converted_prefs)

        conv_rp_arr = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])
        # find GRP, returns np.array
        grpmm, grpmm_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin")
        print("blö", grpmm)
        GRP = cip - grpmm

        # find GRP_ext, returns np.array
        # grpmm_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_ext")
        # GRP_ext = cip - grpmm_ext

        # find GRP, returns np.array
        grpcones, grp_cones_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_cones")
        GRP_cones = cip-grpcones

        # grpcones_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_cones_ext")
        # GRP_cones_ext = cip - grpcones_ext

        # meanGRP_arr = np.mean(conv_rp_arr, axis=0) # projected does not make sense as no constraints
        meanGRP_arr = np.mean(rp_arr, axis=0)
        print(meanGRP_arr)

        # for plotting
        # meanGRP_arr = cip - meanGRP_arr
    else:
        # find GRP, returns np.array
        grpmm = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin")
        GRP = grpmm

        # find GRP_ext, returns np.array
        grpmm_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_ext")
        GRP_ext = grpmm_ext

        # find GRP, returns np.array
        grpcones = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones")
        GRP_cones = grpcones

        grpcones_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones_ext")
        GRP_cones_ext = grpcones_ext

        # JUST "NORMAL" mean TODO: use nautili mean, DUNNO IF THIS IS CORRECT WHATEVER
        imprs = []
        for i in range(len(rp_arr)):
            imprs.append(cip - rp_arr[i])
        meanGRP_arr = np.mean(imprs, axis=0)
        print(meanGRP_arr)

        # for plotting
        meanGRP_arr = cip - meanGRP_arr
    # GRP_mean = {"f_1": meanGRP_arr[0], "f_2": meanGRP_arr[1]}

    grp_name_list = ["GRP mean", "GRP mm", "GRP mm ext", "GRP cones", "GRP cones ext"]

    # create a list of points
    points = []
    points.append(ideal)
    points.append(cip)

    for i in range(len(rp_arr)):
        points.append(rp_arr[i])

    points.append(meanGRP_arr)
    points.append(GRP)
    points.append(GRP_ext)
    points.append(GRP_cones)
    points.append(GRP_cones_ext)

    column_names = []
    column_names.append("ideal")
    column_names.append("cip")

    for i in range(len(rps)):
        column_names.append(f"DM{i+1} RP")

    for i in range(len(grp_name_list)):
        column_names.append(grp_name_list[i])

    print(points)
    import pandas as pd
    # TOOD: make for any number of objs
    # df = pd.DataFrame(points, columns=["f_1", "f_2", "f_3", "f_4"])
    df = pd.DataFrame(points, columns=["f_1", "f_2"])
    # df = df.rename(columns={'0': "f_1", '1': "f_2"})
    df.index = column_names

    print(df)
    # df.to_csv(f"/home/jp/tyot/mop/papers/prefagg_concept/exptables/{name}.csv")
    # fig.write_image(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/paperpics/{name}.pdf", width=800, height=800)

    marker_size = 10

    # TODO: figure out the marker styles better
    # fig = px.scatter(df, x="f_1", y="f_2", color=column_names)

    df_empty = {
        "f_1": [0],
        "f_2": [0],
    }
    fig = px.scatter(df_empty, x="f_1", y="f_2")

    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=marker_size, symbol="circle"))

    # PLOT GRP
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="GRP_mean", showlegend=True, marker=dict(size=marker_size, symbol="square"))
    # additive
    fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="GRP_mm_ext", showlegend=True, marker=dict(size=marker_size, symbol="x"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="GRP_mm", showlegend=True, marker=dict(size=marker_size, symbol="diamond-tall"))
    # cones
    fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="GRP-cones_ext",
                    showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="GRP-cones", showlegend=True, marker=dict(size=marker_size, symbol="diamond-wide"))

    # lines to show the polyhedron
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # FOR saving as pdf
    fig.update_layout(autosize=True, width=800, height=800, title=f"{name}")
    """ for bottom right legend
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    """
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    # fig.update_xaxes(range=[0.5, 1.025])
    # fig.update_yaxes(range=[0.5, 1.025])
    # fig.update_layout(margin_r=300, legend_x=1.2, autosize=True, width=800, height=800, title=f"{name}")
    fig.write_image(f"{pdf_path}{name}.pdf", width=800, height=800)
    fig.write_html(f"{html_path}{name}.html")

    fig.show()


def list_of_dicts_to_array(list_of_dicts):
    """
    Converts a list of dictionaries to a NumPy array, handling an arbitrary number of columns.

    Args:
        list_of_dicts: A list of dictionaries, where each dictionary has keys like 'f_1', 'f_2', etc.

    Returns:
        A NumPy array representing the data.
    """

    if not list_of_dicts:
        return np.array([])  # Handle empty list case

    keys = sorted(list_of_dicts[0].keys())  # Get sorted keys (f_1, f_2, ...)
    array_data = []

    for item in list_of_dicts:
        row = [item[key] for key in keys]
        array_data.append(row)

    return np.array(array_data)


# TODO;: make a visu that only uses additive, cones and po cones, as they are the ones that are differnet.
# MAybe version where it is to decide wheter use original rps or conv rps as constraints??
def solve_test_setting_only3(problem: Problem, rps: list[dict[str, float]], cip, name, proj=False, solver=None, solver_options=None):
    # nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    k = len(ideal)
    q = len(rps)

    rp_arr = list_of_dicts_to_array(rps)
    cip_dict = numpy_array_to_objective_dict(problem, cip)

    GRP = None
    GRP_cones = None
    conv_rp_arr = None

    if proj:
        converted_prefs = []
        # TODO: find PO for each DM RP, for now only for asf, plotting mess
        for rp in range(len(rp_arr)):
            dm_rp = cip - rp_arr[rp]  # convert to improvement direction
            # should convert numpy array (rp) of dm to dict.
            dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
            # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
            p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
                problem,
                symbol=f"asf{rp}",
                reference_point=cip_dict,
                weights=dm_rp,
                reference_point_aug=cip_dict,
            )
            # for 1 RP
            solver = PyomoIpoptSolver(p)
            res = solver.solve(target)
            fs = res.optimal_objectives
            # print(fs)
            converted_prefs.append(fs)

        print(converted_prefs)
        conv_rp_arr = list_of_dicts_to_array(converted_prefs)

        # find GRP, returns np.array
        grpmm, add_s = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin")
        GRP = cip - grpmm

        # find GRP, returns np.array
        grpcones, cones_s = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_cones")
        GRP_cones = cip - grpcones
        # meanGRP_arr = np.mean(conv_rp_arr, axis=0) # projected does not make sense as no constraints
        meanGRP_arr = np.mean(rp_arr, axis=0)

    else:
        # find GRP, returns np.array
        grpmm, add_s = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin")
        GRP = grpmm

        # find GRP, returns np.array
        grpcones, cones_s = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones")
        GRP_cones = grpcones

        # JUST "NORMAL" mean TODO: use nautili mean, DUNNO IF THIS IS CORRECT WHATEVER
        imprs = []
        for i in range(len(rp_arr)):
            imprs.append(cip - rp_arr[i])
        meanGRP_arr = np.mean(imprs, axis=0)
        print(meanGRP_arr)

        # for plotting
        meanGRP_arr = cip - meanGRP_arr

    # Save results as .csv
    grp_name_list = ["meanGRP", "addGRP", "conesGRP"]

    # create a list of points
    points = []
    points.append(ideal)
    points.append(cip)

    for i in range(len(rp_arr)):
        points.append(rp_arr[i])
    points.append(meanGRP_arr)
    points.append(grpmm)
    points.append(grpcones)

    column_names = []
    column_names.append("ideal")
    column_names.append("cip")

    dm_indices = [f" DM{i+1} " for i in range(len(rps))]
    for i in range(len(rps)):
        column_names.append(dm_indices[i])

    for i in range(len(grp_name_list)):
        column_names.append(grp_name_list[i])

    print(points)
    import pandas as pd

    objective_columns = []
    for obj in problem.objectives:
        objective_columns.append(obj.symbol)
    df = pd.DataFrame(points, columns=objective_columns)
    df.index = column_names
    print(df)
    df.to_csv(f"{csv_path}{name}.csv")

    s_df = pd.DataFrame({'add': add_s, 'cones': cones_s}, index=dm_indices)
    s_df.to_csv(f"{csv_path}{name}_s_values.csv")
    print(s_df)

    # TODO: under construction. NEed to add conv. prefs too
    visualize(problem.name, df, column_names, proj=proj)

    # visualize
    marker_size = 15

    if proj:
        problem_pymoo = get_problem(problem.name)
        # problem_pymoo = get_problem(problem.name, n_var=30, n_obj=2)
        pf = problem_pymoo.pareto_front()

        keys = ["f_1", "f_2"]
        # Convert NumPy array to a list of dictionaries
        PF = [dict(zip(keys, row)) for row in pf]
        # plot PF
        if problem.name == "zdt3":
            fig = px.scatter(PF, x="f_1", y="f_2")
        else:
            fig = px.line(PF, x="f_1", y="f_2")

    else:
        df_empty = {
            "f_1": [cip[0]],
            "f_2": [cip[1]],
        }
        fig = px.scatter(df_empty, x="f_1", y="f_2")

    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        if proj:
            fig.add_scatter(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], mode="markers", name=f"proj. DM{
                            i+1}_RP", showlegend=True, marker=dict(size=10, symbol="circle"))
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=marker_size, symbol="circle"))

    # PLOT GRP
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="meanGRP", showlegend=True, marker=dict(size=marker_size, symbol="square"))
    # additive
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="addGRP", showlegend=True, marker=dict(size=marker_size, symbol="x"))
    # cones
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="conesGRP", showlegend=True, marker=dict(size=marker_size, symbol="cross"))

    # lines to show the polyhedron
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # FOR saving as pdf
    fig.update_layout(autosize=True, width=800, height=800, title=f"{name}")
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    """
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    """  # for bottom right legend
    # fig.update_xaxes(range=[0.1, 0.8])
    # fig.update_yaxes(range=[0.1, 0.8])
    # fig.update_layout(margin_r=300, legend_x=1.2, autosize=True, width=800, height=800, title=f"{name}")
    fig.write_image(f"{pdf_path}{name}.pdf", width=800, height=800)
    fig.write_html(f"{html_path}{name}.html")

    fig.show()


def solve_test_setting_old(problem: Problem, rps: list[dict[str, float]], cip, name, solver=None, solver_options=None):
    # nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    k = len(ideal)
    q = len(rps)

    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])

    # find GRP, returns np.array
    grpmm = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}

    # find GRP, returns np.array
    grpcones = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

    grpcones_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones_ext")
    # improvmenet direction
    GRP_cones_ext = cip - grpcones_ext
    # make dict from the GPR array
    GRP_cones_ext = {"f_1": GRP_cones_ext[0], "f_2": GRP_cones_ext[1]}

    # JUST "NORMAL" mean TODO: use nautili mean, DUNNO IF THIS IS CORRECT WHATEVER
    imprs = []
    for i in range(len(rp_arr)):
        imprs.append(cip - rp_arr[i])
    meanGRP_arr = np.mean(imprs, axis=0)
    print(meanGRP_arr)

    # for plotting
    meanGRP_arr = cip - meanGRP_arr

    df = {
        "f_1": [0],
        "f_2": [0],
    }

    marker_size = 10
    # TODO: figure out the marker styles better
    fig = px.scatter(df, x="f_1", y="f_2")
    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=marker_size, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=marker_size, symbol="circle"))

    # PLOT GRP
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="GRP_mean", showlegend=True, marker=dict(size=marker_size, symbol="square"))
    # additive
    fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="GRP_mm_ext", showlegend=True, marker=dict(size=marker_size, symbol="x"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="GRP_mm", showlegend=True, marker=dict(size=marker_size, symbol="diamond-tall"))
    # cones
    fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="GRP-cones_ext",
                    showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="GRP-cones", showlegend=True, marker=dict(size=marker_size, symbol="diamond-wide"))

    # lines to show the polyhedron
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # FOR saving as pdf
    fig.update_layout(autosize=True, width=800, height=800, title=f"{name}")

    """ for bottom right legend
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    """
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    # fig.update_layout(margin_r=300, legend_x=1.2, autosize=True, width=800, height=800, title=f"{name}")
    # fig.write_image(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/paperpics/{name}.pdf", width=800, height=800)
    # fig.write_html(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/paperpics/htmls/{name}.html")

    # fig.show()


def agg_rps_test_mm(problem: Problem, rps: list[dict[str, float]], cip, name, solver=None, solver_options=None):

    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # cip_dict = get_nadir_dict(problem)
    # for maxmins
    # cip = nadir

    cip = cip
    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    k = len(ideal)
    q = len(rps)

    all_prefs = rps  # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    # cip = np.array(np.amax(rp_arr, axis=0))
    print(cip)

    # find GRP, returns np.array
    grpmm = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}

    # find GRP, returns np.array
    grpcones = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

    grpcones_ext = find_GRP(rp_arr, cip, k, q, ideal, rp_arr, "maxmin_cones_ext")
    # improvmenet direction
    GRP_cones_ext = cip - grpcones_ext
    # make dict from the GPR array
    GRP_cones_ext = {"f_1": GRP_cones_ext[0], "f_2": GRP_cones_ext[1]}

    # mean_PO =

    # JUST "NORMAL" mean TODO: use nautili mean, DUNNO IF THIS IS CORRECT WHATEVER
    imprs = []
    for i in range(len(rp_arr)):
        imprs.append(cip - rp_arr[i])
    meanGRP_arr = np.mean(imprs, axis=0)
    print(meanGRP_arr)
    meanGRP = numpy_array_to_objective_dict(problem, meanGRP_arr)

    # for plotting
    meanGRP_arr = cip - meanGRP_arr

    """
    SOLVING
    """
    init_solver = guess_best_solver(problem) if solver is None else solver
    print("solver to use", init_solver)
    _solver_options = None if solver_options is None or solver is None else solver_options

    # use nondiff
    if problem.name == "zdt3":

        rho = 0.00001
        # TODO: MEAN
        problem_w_asf3, target3 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=meanGRP,
            reference_point_aug=cip_dict,
            rho=rho
        )
        """
        problem_w_asf3, target3 = add_guess_sf_diff(
            problem,
            symbol="guess",
            reference_point=meanGRP,
            # weights=meanGRP,
            # reference_point_aug=cip_dict,
            rho=rho
        )
        problem_w_asf3, target3 = add_stom_sf_diff(
            problem,
            symbol="stom",
            reference_point=meanGRP,
            # weights=meanGRP,
            # reference_point_aug=cip_dict,
            rho=rho
        )
        """
        from desdeo.tools import GurobipySolver

        solver3 = GurobipySolver(problem_w_asf3)
        res3 = solver3.solve(target3)
        xs3 = res3.optimal_variables
        fs_mean = res3.optimal_objectives
        print("final solution from mean", fs_mean)

        problem_w_asf, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP,
            reference_point_aug=cip_dict,
            rho=rho
            # rho=0.0001
        )
        solver = GurobipySolver(problem_w_asf)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs_mm = res.optimal_objectives
        print("final solution from maxmin", fs_mm)

        problem_w_asf4, target4 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP_ext,
            reference_point_aug=cip_dict,
            rho=rho
            # rho=0.0001
        )
        solver4 = GurobipySolver(problem_w_asf4)
        res4 = solver4.solve(target4)
        xs4 = res4.optimal_variables
        fs_mm_ext = res4.optimal_objectives
        print("final solution from maxmin ext", fs_mm_ext)

        # solve with maxmincones
        problem_w_asf2, target2 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP_cones,
            reference_point_aug=cip_dict,
            rho=rho
            # rho=0.001
        )
        solver2 = GurobipySolver(problem_w_asf2)
        # solver2 = NevergradGenericSolver(problem_w_asf2) # NEVERGRAD IS SO ASS
        # solver2 = init_solver(problem_w_asf2)
        res2 = solver2.solve(target2)
        xs2 = res2.optimal_variables
        fs_cones = res2.optimal_objectives
        print("final solution from maxmin-cones", fs_cones)

    else:

        # TODO: MEAN
        problem_w_asf3, target3 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=meanGRP,
            reference_point_aug=cip_dict,
            # rho=0.0001
        )
        # solver3 = PyomoIpoptSolver(problem_w_asf3)
        solver3 = init_solver(problem_w_asf3)
        res3 = solver3.solve(target3)
        xs3 = res3.optimal_variables
        fs_mean = res3.optimal_objectives
        print("final solution from mean", fs_mean)

        problem_w_asf, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP,
            reference_point_aug=cip_dict,
            # rho=0.0001
        )
        solver = PyomoIpoptSolver(problem_w_asf)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs_mm = res.optimal_objectives
        print("final solution from maxmin", fs_mm)

        problem_w_asf4, target4 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP_ext,
            reference_point_aug=cip_dict,
            # rho=0.0001
        )
        solver4 = PyomoIpoptSolver(problem_w_asf4)
        res4 = solver4.solve(target4)
        xs4 = res4.optimal_variables
        fs_mm_ext = res4.optimal_objectives
        print("final solution from maxmin ext", fs_mm_ext)

        # solve with maxmincones
        problem_w_asf2, target2 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP_cones,
            reference_point_aug=cip_dict,
            # rho=0.001
        )
        solver2 = PyomoIpoptSolver(problem_w_asf2)
        # solver2 = NevergradGenericSolver(problem_w_asf2) # NEVERGRAD IS SO ASS
        # solver2 = init_solver(problem_w_asf2)
        res2 = solver2.solve(target2)
        xs2 = res2.optimal_variables
        fs_cones = res2.optimal_objectives
        print("final solution from maxmin-cones", fs_cones)

        # solve with maxmincones_ext
        problem_w_asf22, target22 = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=GRP_cones_ext,
            reference_point_aug=cip_dict,
            # rho=0.001
        )
        solver22 = PyomoIpoptSolver(problem_w_asf22)
        # solver2 = NevergradGenericSolver(problem_w_asf2) # NEVERGRAD IS SO ASS
        # solver2 = init_solver(problem_w_asf2)
        res22 = solver22.solve(target22)
        xs22 = res22.optimal_variables
        fs_cones_ext = res22.optimal_objectives
        print("final solution from maxmin-cones_ext", fs_cones_ext)

    """
    VISUALIZING
    """
    solutions = [fs_mean, fs_mm, fs_mm_ext, fs_cones, fs_cones_ext]

    keys = ["f_1", "f_2"]
    namelist = ["PO_mean", "PO_mm", "PO_mm_ext", "PO_cones", "PO_cones_ext"]

    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }

    all_solutions = pd.DataFrame(all_solutions, columns=["f_1", "f_2", "names"])

    problem_pymoo = get_problem(problem.name)
    pf = problem_pymoo.pareto_front()

    # Convert NumPy array to a list of dictionaries
    PF = [dict(zip(keys, row)) for row in pf]
    # plot PF
    if problem.name == "zdt3":
        fig = px.scatter(PF, x="f_1", y="f_2")
    else:
        fig = px.line(PF, x="f_1", y="f_2")

    # TODO: figure out the marker styles better
    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=15, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=15, symbol="x"))

    # PLOT GRP
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="GRP_mean", showlegend=True, marker=dict(size=15, symbol="square"))
    fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="GRP_mm_ext", showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="GRP_mm", showlegend=True, marker=dict(size=15, symbol="diamond-tall"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="GRP-cones", showlegend=True, marker=dict(size=15, symbol="diamond-wide"))
    fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="GRP-cones_ext",
                    showlegend=True, marker=dict(size=15, symbol="cross"))
    # fig.update_traces(marker=dict(size=15, symbol="x"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15))
    # PLOT results
    # fig.add_scatter(all_solutions,  x=all_solutions.f_1, y=all_solutions.f_2, mode="markers", name=all_solutions.names ,showlegend=True)
    for i in range(len(namelist)):
        fig.add_scatter(x=[all_solutions["f_1"][i]], y=[all_solutions["f_2"][i]], mode="markers",
                        name=all_solutions.names[i], showlegend=True, marker=dict(size=15, symbol="circle"))

    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_traces(all_solutions)
    # fig.update_traces(marker=dict(size=15))
    # fig.update_traces(marker=dict(size=15, symbol="x"))

    # lines to show the polyhedron
    # fig.add_scatter(x=[0.2, 0.45, 0.55, 0.2], y=[0.4, 0.4, 0.1, 0.4], mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # FOR saving as pdf
    fig.update_layout(autosize=False, width=800, height=800, title=f"{name}")
    fig.write_image(f"{pdf_path}{name}.pdf", width=800, height=800)
    fig.write_html(f"{html_path}{name}.html")

    fig.show()


def agg_rps_test_eq(problem: Problem, rps: list[dict[str, float]], cip, name):

    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    cip = cip
    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    k = len(ideal)
    q = len(rps)

    all_prefs = rps  # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    # cip = np.array(np.amax(rp_arr, axis=0))
    print(cip)

    # EQ VARIANTS
    # SOLVE FOR ALL DMS

    converted_prefs = []
    # TODO: find PO for each DM RP, for now only for asf, plotting mess
    for rp in range(len(rp_arr)):
        dm_rp = cip - rp_arr[rp]  # convert to improvement direction
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
        # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
        p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol=f"asf{rp}",
            reference_point=cip_dict,
            weights=dm_rp,
            reference_point_aug=cip_dict,
        )
        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    print(converted_prefs)

    conv_rp_arr = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])

    # TODO: send also the og rps
    # find GRP, returns np.array
    grpmm, grpmm_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}
    print(f"maxmin {GRP}")

    # find GRP_ext, returns np.array
    grpmm_ext, grpext_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}
    print(f"maxmin ext {GRP_ext}")

    # find GRP, returns np.array
    grpcones, grp_cones_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}
    print(f"maxmin cones {GRP_cones}")

    # find GRP, returns np.array
    grpcones_ext, grp_cones_ext_s_values = find_GRP(conv_rp_arr, cip, k, q, ideal, rp_arr, "eq_maxmin_cones_ext")
    # improvmenet direction
    GRP_cones_ext = cip - grpcones_ext
    # make dict from the GPR array
    GRP_cones_ext = {"f_1": GRP_cones_ext[0], "f_2": GRP_cones_ext[1]}
    print(f"maxmin cones ext {GRP_cones_ext}")

    grpmean = np.mean(rp_arr, axis=0)
    # grpmean = cip - np.mean(rp_arr, axis=0)
    GRPmean = {"f_1": grpmean[0], "f_2": grpmean[1]}
    print(f"mean {GRPmean}")

    print("S VALUES")
    print(grpmm_s_values)
    print(grp_cones_s_values)

    """
    SOLVING
    """

    problem_w_asf5, target5 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRPmean,
        reference_point_aug=cip_dict,
    )
    solver5 = PyomoIpoptSolver(problem_w_asf5)
    res5 = solver5.solve(target5)
    fs_mean = res5.optimal_objectives
    print("final solution from mean", fs_mean)

    problem_w_asf, target = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP,
        reference_point_aug=cip_dict,
    )
    solver = PyomoIpoptSolver(problem_w_asf)
    res = solver.solve(target)
    fs_mm = res.optimal_objectives
    print("final solution from maxmin", fs_mm)

    problem_w_asf4, target4 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_ext,
        reference_point_aug=cip_dict,
    )
    solver4 = PyomoIpoptSolver(problem_w_asf4)
    res4 = solver4.solve(target4)
    fs_mm_ext = res4.optimal_objectives
    print("final solution from maxmin ext", fs_mm_ext)

    # solve with maxmincones
    problem_w_asf2, target2 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_cones,
        reference_point_aug=cip_dict,
    )
    solver2 = PyomoIpoptSolver(problem_w_asf2)
    res2 = solver2.solve(target2)
    fs_cones = res2.optimal_objectives
    print("final solution from maxmin cones", fs_cones)

    # solve with maxmincones_ext
    problem_w_asf22, target22 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_cones_ext,
        reference_point_aug=cip_dict,
    )
    solver22 = PyomoIpoptSolver(problem_w_asf22)
    res22 = solver22.solve(target22)
    fs_cones_ext = res22.optimal_objectives
    print("final solution from maxmin cones ext", fs_cones_ext)
    """
    VISUALIZING
    """

    solutions = [fs_mean, fs_mm, fs_mm_ext, fs_cones, fs_cones_ext]

    keys = ["f_1", "f_2"]
    namelist = ["PO_mean", "PO_mm", "PO_mm_ext", "PO_cones", "PO_cones_ext"]

    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }

    all_solutions = pd.DataFrame(all_solutions, columns=["f_1", "f_2", "names"])

    problem_pymoo = get_problem(problem.name, n_var=30, n_obj=2)

    pf = problem_pymoo.pareto_front()

    # Convert NumPy array to a list of dictionaries
    PF = [dict(zip(keys, row)) for row in pf]

    # plot PF
    if problem.name == "zdt3":  # or problem.name == "dtlz2":
        fig = px.scatter(PF, x="f_1", y="f_2")
    else:
        fig = px.line(PF, x="f_1", y="f_2")

    # TODO: figure out the marker styles better
    fig.add_scatter(x=[cip[0]], y=[cip[1]], mode="markers", name="CIP", showlegend=True, marker=dict(size=15, symbol="star"))
    fig.add_scatter(x=[ideal[0]], y=[ideal[1]], mode="markers", name="ideal", showlegend=True, marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        fig.add_scatter(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], mode="markers",
                        name=f"proj. DM{i+1}_RP", showlegend=True, marker=dict(size=10, symbol="circle"))
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{
                        i+1}_RP", showlegend=True, marker=dict(size=10, symbol="circle"))

    # PLOT GRP
    fig.add_scatter(x=[grpmean[0]], y=[grpmean[1]], mode="markers", name="GRP-mean", showlegend=True, marker=dict(size=15, symbol="pentagon"))
    # fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="PO-GRP_mm_ext", showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="x"))
    # fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="PO-GRP-cones_ext", showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="PO-GRP-cones", showlegend=True, marker=dict(size=15, symbol="cross"))

    # fig.update_traces(marker=dict(size=15, symbol="x"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15))
    # PLOT results
    # fig.add_scatter(all_solutions,  x=all_solutions.f_1, y=all_solutions.f_2, mode="markers", name=all_solutions.names ,showlegend=True)
    # for i in range(len(namelist)):
    # fig.add_scatter(x=[all_solutions["f_1"][i]], y=[all_solutions["f_2"][i]], mode="markers",
    #                name=all_solutions.names[i], showlegend=True, marker=dict(size=15, symbol="circle"))

    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_traces(all_solutions)
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # fig.update_traces(marker=dict(size=15, symbol="x"))

    # lines to show the polyhedron
    # fig.add_scatter(x=[0.2, 0.45, 0.55, 0.2], y=[0.4, 0.4, 0.1, 0.4], mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    fig.add_scatter(x=create_line_path(rp_arr[:, 0]), y=create_line_path(rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    # fig.add_scatter(x=create_line_path(conv_rp_arr[:, 0]), y=create_line_path(conv_rp_arr[:, 1]),
    #                mode="lines", line=dict(color="#803080"), name="valid_area", showlegend=True)

    # FOR saving as pdf
    fig.update_layout(autosize=False, width=800, height=800, title=f"{name}")
    fig.write_image(f"{pdf_path}{name}.pdf", width=800, height=800)
    fig.write_html(f"{html_path}{name}.html")

    fig.show()

def agg_rps_test_eq_3d(problem: Problem, rps: list[dict[str, float]], cip, name):

    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    cip = cip
    cip_dict = {"f_1": cip[0], "f_2": cip[1], "f_3": cip[2]}

    k = len(ideal)
    q = len(rps)

    all_prefs = rps  # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"], col["f_3"]] for col in rps])
    # cip = np.array(np.amax(rp_arr, axis=0))
    print(cip)

    # EQ VARIANTS
    # SOLVE FOR ALL DMS

    converted_prefs = []
    # TODO: find PO for each DM RP, for now only for asf, plotting mess
    for rp in range(len(rp_arr)):
        dm_rp = cip - rp_arr[rp]  # convert to improvement direction
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
        # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
        p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol=f"asf{rp}",
            reference_point=cip_dict,
            weights=dm_rp,
            reference_point_aug=cip_dict,
        )
        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    print(converted_prefs)

    conv_rp_arr = np.array([[col["f_1"], col["f_2"], col["f_3"]] for col in converted_prefs])

    # find GRP, returns np.array
    grpmm, add_s = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1], "f_3": GRP[2]}

    # find GRP_ext, returns np.array
    grpmm_ext, add_s_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1], "f_3": GRP_ext[2]}

    # find GRP, returns np.array
    grpcones, cones_s = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1], "f_3": GRP_cones[2]}

    grpmean = cip - np.mean(conv_rp_arr, axis=0)
    GRPmean = {"f_1": grpmean[0], "f_2": grpmean[1], "f_3": grpmean[2]}

    """
    SOLVING
    """

    problem_w_asf5, target5 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRPmean,
        reference_point_aug=cip_dict,
    )
    solver5 = PyomoIpoptSolver(problem_w_asf5)
    res5 = solver5.solve(target5)
    xs = res5.optimal_variables
    fs_mean = res5.optimal_objectives
    print("final solution from mean", fs_mean)

    problem_w_asf, target = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP,
        reference_point_aug=cip_dict,
    )
    solver = PyomoIpoptSolver(problem_w_asf)
    res = solver.solve(target)
    xs = res.optimal_variables
    fs_mm = res.optimal_objectives
    print("final solution from maxmin", fs_mm)

    problem_w_asf4, target4 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_ext,
        reference_point_aug=cip_dict,
    )
    solver4 = PyomoIpoptSolver(problem_w_asf4)
    res4 = solver4.solve(target4)
    xs4 = res4.optimal_variables
    fs_mm_ext = res4.optimal_objectives
    print("final solution from maxmin ext", fs_mm_ext)

    # solve with maxmincones
    problem_w_asf2, target2 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_cones,
        reference_point_aug=cip_dict,
    )
    solver2 = PyomoIpoptSolver(problem_w_asf2)
    res2 = solver2.solve(target2)
    xs2 = res2.optimal_variables
    fs_cones = res2.optimal_objectives
    print("final solution from maxmin cones", fs_cones)

    """
    VISUALIZING
    """

    solutions = [fs_mean, fs_mm, fs_mm_ext, fs_cones]

    keys = ["f_1", "f_2"]
    namelist = ["PO_mean", "PO_mm", "PO_mm_ext", "PO_cones"]

    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "f_3": [s["f_3"] for s in solutions],
        "names": namelist,
    }

    all_solutions = pd.DataFrame(all_solutions, columns=["f_1", "f_2", "f_3", "names"])

    problem_pymoo = get_problem(problem.name)
    pf = problem_pymoo.pareto_front()

    # Convert NumPy array to a list of dictionaries
    PF = [dict(zip(keys, row)) for row in pf]
    # plot PF
    if problem.name == "zdt3":
        fig = px.scatter(PF, x="f_1", y="f_2")
    if problem.name == "dltz2":
        fig = px.scatter3d(PF, x="f_1", y="f_2", z="f_3")
    else:
        fig = px.line(PF, x="f_1", y="f_2")

    # TODO: figure out the marker styles better
    fig.add_scatter3d(x=[cip[0]], y=[cip[1]], z=[cip[2]], mode="markers", name="CIP", showlegend=True, marker=dict(size=15, symbol="star"))
    fig.add_scatter3d(x=[ideal[0]], y=[ideal[1]], z=[ideal[2]], mode="markers", name="ideal", showlegend=True, marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))

    # plot RPs
    for i in range(len(rp_arr)):
        # fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=10, symbol="x"))
        fig.add_scatter3d(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], z=[conv_rp_arr[i][2]], mode="markers", name=f"converted DM{
            i+1}_RP", showlegend=True, marker=dict(size=15, symbol="x"))

    # PLOT GRP
    # fig.add_scatter(x=[grpmean[0]], y=[grpmean[1]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="pentagon"))
    fig.add_scatter3d(x=[grpmm[0]], y=[grpmm[1]], z=[grpmm[2]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="diamond-tall"))
    fig.add_scatter3d(x=[grpmm_ext[0]], y=[grpmm_ext[1]], z=[grpmm_ext[2]], mode="markers",
                      name="PO-GRP_mm_ext", showlegend=True, marker=dict(size=15, symbol="diamond-wide"))
    fig.add_scatter3d(x=[grpcones[0]], y=[grpcones[1]], z=[grpcones[2]], mode="markers",
                      name="PO-GRP-cones", showlegend=True, marker=dict(size=15, symbol="cross"))
    # fig.update_traces(marker=dict(size=15, symbol="x"))
    # fig.update_traces(marker=dict(size=15, symbol="star"))
    # fig.update_traces(marker=dict(size=15))
    # PLOT results
    # fig.add_scatter(all_solutions,  x=all_solutions.f_1, y=all_solutions.f_2, mode="markers", name=all_solutions.names ,showlegend=True)
    for i in range(len(namelist)):
        fig.add_scatter3d(x=[all_solutions["f_1"][i]], y=[all_solutions["f_2"][i]], z=[all_solutions["f_3"][i]], mode="markers",
                          name=all_solutions.names[i], showlegend=True, marker=dict(size=15, symbol="circle"))

    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    # fig.add_traces(all_solutions)
    # fig.update_traces(marker=dict(size=15))
    # fig.update_traces(marker=dict(size=15, symbol="x"))

    # lines to show the polyhedron
    # fig.add_scatter(x=[0.2, 0.45, 0.55, 0.2], y=[0.4, 0.4, 0.1, 0.4], mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    fig.add_scatter3d(x=create_line_path(conv_rp_arr[:, 0]), y=create_line_path(conv_rp_arr[:, 1], z=create_line_path(conv_rp_arr[:, 2]),),
                      mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)

   # fig.write_image(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/misc/{name}.pdf", width=600, height=600)

    fig.show()

def create_line_path(arr):
    return np.append(arr, arr[0])


def experiment_optDM1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.8, "f_2": 0.6},
        {"f_1": 0.8, "f_2": 0.0},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment_optDM2(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.8, "f_2": 0.6},
        {"f_1": 0.8, "f_2": 0.0},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)


def experiment_zdt3(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt3(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": -1},
        {"f_1": 0.8, "f_2": 0.6},
        {"f_1": 0.8, "f_2": 0.0},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment_solution_process(case_name):
    # n_variables = 30
    # n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    from desdeo.problem.testproblems import binh_and_korn, re22
    # problem = binh_and_korn()
    problem = re22()

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.7, "f_2": 0.5},
        {"f_1": 0.2, "f_2": 0.8},
        {"f_1": 0.8937, "f_2": 0.1063},  # 0.99
    ]
    cip = np.array([5, 2])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)


def experiment1_solution_process1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.7, "f_2": 0.5},
        {"f_1": 0.2, "f_2": 0.8},
        {"f_1": 0.8937, "f_2": 0.1063},  # 0.99
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment1_solution_process2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.4, "f_2": 0.5},
        {"f_1": 0.6, "f_2": 0.6},
        {"f_1": 0.6, "f_2": 0.3},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([0.7, .7])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment1_solution_process3(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.4, "f_2": 0.45},
        {"f_1": 0.5, "f_2": 0.5},
        {"f_1": 0.5, "f_2": 0.3},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([0.6, .6])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name)
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment2_solution_process1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.4, "f_2": 0.8},
        {"f_1": 0.7, "f_2": 0.7},
        {"f_1": 0.6, "f_2": 0.5},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment1_scaling(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.1, "f_2": 0.95},
        {"f_1": 0.5, "f_2": 0.83},
        {"f_1": 0.99, "f_2": 0.65},
        {"f_1": 0.6, "f_2": 0.75},
    ]
    # rp3 f2 == rp4 f2, a > 90
    # eli rp3 f2 < rp4 f2 että mahd. a < 90
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, False)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_scaling(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    problem = dtlz2(n_variables, n_objectives)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.1, "f_2": 0.95},
        {"f_1": 0.5, "f_2": 0.83},
        {"f_1": 0.9, "f_2": 0.692},
        {"f_1": 0.6, "f_2": 0.75},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name, True)
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_scaling2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.1, "f_2": 0.95},
        {"f_1": 0.5, "f_2": 0.83},
        {"f_1": 0.4, "f_2": 0.81},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_solution_process3(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.4, "f_2": 0.45},
        {"f_1": 0.5, "f_2": 0.5},
        {"f_1": 0.5, "f_2": 0.3},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([0.6, .6])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name, True)
    agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_test_river_poll(case_name):
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    from desdeo.problem.testproblems import river_pollution_problem
    problem = river_pollution_problem(five_objective_variant=False)

    nadir = get_nadir_dict(problem)
    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {
            "f_1": 5,
            "f_2": 3.01,
            "f_3": 0.6,
            "f_4": -1
        },
        {
            "f_1": 4.85,
            "f_2": 2.91,
            "f_3": 1.4,
            "f_4": -2
        }, {
            "f_1": 5.1,
            "f_2": 2.90,
            "f_3": 1.7,
            "f_4": -4.5
        },
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    # cip = np.array([1, 1])
    cip = objective_dict_to_numpy_array(problem, nadir)
    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment3_test(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 3
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    problem = zdt2(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.9, "f_2": 0.11},
        {"f_1": 0.9, "f_2": 0.},
        {"f_1": 0.39, "f_2": 0.5},
        {"f_1": 0.0, "f_2": 0.9},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment2_test_dtlz2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 3
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    problem = dtlz2(n_objectives, n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.9, "f_2": 0., "f_3": 0.5},
        {"f_1": 0.0, "f_2": 0.9, "f_3": 0.7},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([1, 1, 1])

    # agg_rps_test
    agg_rps_test_eq_3d(problem, reference_points, cip, case_name)

def experiment1_pessmistic1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.5, "f_2": 0.7},
        {"f_1": 0.45, "f_2": 0.5},
        {"f_1": 0.5, "f_2": 0.45},
    ]
    cip = np.array([0.8, 0.8])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment1_zdt2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.8, "f_2": 0.5},
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": 0.8},
    ]
    cip = np.array([0.9, 0.9])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_optdm1(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.6, "f_2": 0.5},
        {"f_1": 0.9, "f_2": 0.8},
        {"f_1": 0.75, "f_2": 0.85},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_optdm1_2(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.9, "f_2": 0.7},
        {"f_1": 0.7, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name)
    # solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp2_optdm1(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.9, "f_2": 0.7},
        {"f_1": 0.7, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)


def exp3_optdm1(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.9, "f_2": 0.7},
        {"f_1": 0.7, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name, True)
    agg_rps_test_eq(problem, reference_points, cip, case_name)


def exp_vertdms(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.3},
        {"f_1": 0.2, "f_2": 0.5},
        {"f_1": 0.5, "f_2": 0.2},
    ]
    cip = np.array([0.75, 0.75])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name)
    # solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp2_vertdms(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.3},
        {"f_1": 0.2, "f_2": 0.5},
        {"f_1": 0.3, "f_2": 0.7},
    ]
    cip = np.array([0.75, 0.75])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)

def exp1_small_step(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.9, "f_2": 0.85},
        {"f_1": 0.8, "f_2": 0.92},
        {"f_1": 0.77, "f_2": 0.97},
        {"f_1": 0.95, "f_2": 0.82},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, False)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)
    #

def exp2_small_step(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.9, "f_2": 0.85},
        {"f_1": 0.8, "f_2": 0.92},
        {"f_1": 0.77, "f_2": 0.97},
        {"f_1": 0.95, "f_2": 0.82},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting_only3(problem, reference_points, cip, case_name, True)
    # agg_rps_test_eq(problem, reference_points, cip, case_name)


def exp_change1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4},
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.2},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_change2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4},
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.35},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_change3(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4},
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.4},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_shapePF(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.5},
        {"f_1": 0.55, "f_2": 0.1},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_shapePF2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.5},
        {"f_1": 0.55, "f_2": 0.1},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_cip1(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.4},
        {"f_1": 0.55, "f_2": 0.1},
    ]
    cip = np.array([0.8, 0.5])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)


def exp_cip2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.6, "f_2": 0.8},
        {"f_1": 0.66, "f_2": 0.67},
        {"f_1": 0.82, "f_2": 0.72},
        {"f_1": 0.8, "f_2": 0.75},
        {"f_1": 0.62, "f_2": 0.9},

    ]
    cip = np.array([1, 1])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name)
    solve_test_setting_only3(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)


def exp_cip3(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.05},
        {"f_1": 0.4, "f_2": 0.15},
        {"f_1": 0.42, "f_2": 0.39},
        {"f_1": 0.3, "f_2": 0.3},
        {"f_1": 0.1, "f_2": 0.44},
    ]
    cip = np.array([0.5, 0.5])

    # agg_rps_test
    # solve_test_setting(problem, reference_points, cip, case_name)
    solve_test_setting_only3(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)

def test_sm_values(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.05},
        {"f_1": 0.4, "f_2": 0.15},
        {"f_1": 0.42, "f_2": 0.39},
        {"f_1": 0.3, "f_2": 0.3},
        {"f_1": 0.1, "f_2": 0.44},
    ]
    cip = np.array([0.5, 0.5])

    # agg_rps_test
    solve_test_setting(problem, reference_points, cip, case_name)
    # agg_rps_test_mm(problem, reference_points, cip, case_name)


def experiment_new(case_name):

    from desdeo.problem.testproblems import re22, river_pollution_problem

    # sp = river_pollution_problem()

    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))
    # eq_example_optdm1
    """
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.4},
        {"f_1": 0.55, "f_2": 0.1},
    ]
    cip = np.array([0.8, 0.5])
    """
    reference_points = [
        {"f_1": 0.3, "f_2": 0.3},
        {"f_1": 0.4, "f_2": 0.6},
        {"f_1": 0.7, "f_2": 0.7},
    ]

    cip = np.array([1, 1])
    q = 3
    from preference_aggregation import subproblem_linear, subproblem_nl, simple_test_problem2

    sp = simple_test_problem2()

    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in reference_points])
    # rp_arr = objective_dict_to_numpy_array(problem, reference_points)
    # sp = subproblem_linear(rp_arr, cip, n_objectives, q, ideal)
    sp = subproblem_nl(rp_arr, cip, n_objectives, q, ideal)
    print(sp)
    from desdeo.tools import ScipyMinimizeSolver, GurobipySolver, NevergradGenericSolver

    solv_opts = ScipyMinimizeOptions(method="SLSQP")
    solver = ScipyMinimizeSolver(sp, options=solv_opts)
    #solver2 = PyomoIpoptSolver(sp)
    solver_options = NevergradGenericOptions(budget=500, optimizer="CMA")
    solver3 = NevergradGenericSolver(sp, solver_options)

    # gr_solver = GurobipySolver(sp)

    results = solver.solve("f_1")
    #results2 = solver2.solve("f_1")
    results3 = solver3.solve("f_1")#
    # results3 = gr_solver.solve("f_1")
    print("======================")
    print("RESULLTS:  ", results)
    # print("RESULLTS:  ", results2)
    print("RESULLTS:  ", results3)
    print("======================")

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    # TODO find a real problem that works
    # experiment_solution_process("realproblem")
    # exp2_optdm1("e2o2")
    # exp_optdm1_2("e1o1")
    # exp_vertdms("verticaldms_2")
    # experiment2_solution_process1("e2s1")
    # experiment2_scaling("eprojscalealphab")
    # experiment2_scaling("e2dtlz2")
    # experiment2_scaling("e2scalecat")
    # experiment2_scaling2("criticalangle")
    # exp1_small_step("e1small")

    experiment_new("testing")

    """
    print("Running problem")
    print("Running problem")
    exp_optdm1_2("e1o1")
    print("Running problem")
    exp_vertdms("verticaldms")
    print("Running problem")
    experiment1_zdt2("zdt2example")

    print("Running problem")
    experiment1_solution_process1("e1s1")
    print("Running problem")
    experiment1_solution_process2("e1s2")
    print("Running problem")
    experiment1_solution_process3("e1s3")
    print("Running problem")
    experiment1_pessmistic1("e1p1")
    print("Running problem")
    exp_optdm1_2("eo1")

    print("Running problem")
    exp_change1("change1")
    print("Running problem")
    exp_change2("change2")
    print("Running problem")
    exp_change3("change3")
    print("Running problem")
    exp_cip1("cip1")
    """

    # exp_vertdms("verticaldms_2")

    # exp_shapePF("shape1")
    # exp_shapePF2("shape2")
    # experiment_optDM2("eo3")
    # exp_cip2("Example 1 Alt2")
    # exp_cip3("Example 2")

    # EQ variants
    # exp2_optdm1("e2o2")
    # exp2_vertdms("vert2")

    # exp2_small_step("e2small")
    """
        {"f_1": 0.1, "f_2": 0.95},
        {"f_1": 0.5, "f_2": 0.83},
        {"f_1": 0.95, "f_2": 0.7673},
        {"f_1": 0.6, "f_2": 0.75},
    exp3_optdm1("e3o2 alt")
    experiment2_solution_process1("e2s1")
    experiment2_solution_process2("e2s2")
    experiment2_solution_process3("e2s3")

    """
    # test_sm_values("smvalues")
    # Below dtlz2 does not work for some reason.
    #
    # experiment2_test_river_poll("test")
    # experiment_zdt3("ezdt3")  # does not work

    # exp2_optdm1("e2o2")
    # experiment2_test("test")
    # experiment3_test("test")
    # experiment2_("")
