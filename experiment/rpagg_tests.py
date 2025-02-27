from desdeo.problem import Problem, objective_dict_to_numpy_array, numpy_array_to_objective_dict, get_ideal_dict, get_nadir_dict
from desdeo.tools.scalarization import add_asf_diff, add_guess_sf_diff, add_stom_sf_diff, add_asf_generic_diff, add_asf_generic_nondiff, add_guess_sf_nondiff
import numpy as np

from desdeo.problem.testproblems import zdt1, zdt2, zdt3, dtlz2
# noqa

from desdeo.tools.utils import guess_best_solver, PyomoIpoptSolver, NevergradGenericSolver

from preference_aggregation import find_GRP


import matplotlib.pyplot as plt

from pymoo.problems import get_problem
from pymoo.util.plotting import plot

import plotly.express as px

import pandas as pd

import plotly.io as pio
pio.kaleido.scope.mathjax = None

# TODO: only works iwth maxmin ja maxmin-cones. MAke another with eq variants
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
    grpmm = find_GRP(rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP(rp_arr, cip, k, q, ideal, nadir, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}

    # find GRP, returns np.array
    grpcones = find_GRP(rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

    grpcones_ext = find_GRP(rp_arr, cip, k, q, ideal, nadir, "maxmin_cones_ext")
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
    # fig.update_layout(autosize=False, width=800, height=800)
    # fig.write_image(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/paperpics/{name}.pdf", width=600, height=600)
    # fig.write_html(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/paperpics/htmls/{name}.html")

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
        xs = res.optimal_variables
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    """
    for d in all_prefs:

        problem_w_asf, target = add_asf_generic_diff( # use nondiff as default like in nautili
            problem,
            symbol="asf",
            reference_point=cip_dict,
            weights=d,
            reference_point_aug=cip_dict,
         )
        solver = PyomoIpoptSolver(problem_w_asf)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs = res.optimal_objectives
        print(fs)
        converted_prefs.merge(fs)
    """

    print(converted_prefs)

    conv_rp_arr = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])

    # TODO: send also the og rps
    # find GRP, returns np.array
    grpmm = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}

    # find GRP, returns np.array
    grpcones = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

    # find GRP, returns np.array
    grpcones_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_cones_ext")
    # improvmenet direction
    GRP_cones_ext = cip - grpcones_ext
    # make dict from the GPR array
    GRP_cones_ext = {"f_1": GRP_cones_ext[0], "f_2": GRP_cones_ext[1]}

    grpmean = cip - np.mean(conv_rp_arr, axis=0)
    GRPmean = {"f_1": grpmean[0], "f_2": grpmean[1]}

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

    # solve with maxmincones_ext
    problem_w_asf22, target22 = add_asf_generic_diff(  # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=cip_dict,
        weights=GRP_cones_ext,
        reference_point_aug=cip_dict,
    )
    solver22 = PyomoIpoptSolver(problem_w_asf22)
    res22 = solver2.solve(target22)
    xs22 = res22.optimal_variables
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
        # fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=10, symbol="x"))
        fig.add_scatter(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], mode="markers", name=f"converted DM{
                        i+1}_RP", showlegend=True, marker=dict(size=15, symbol="x"))

    # PLOT GRP
    # fig.add_scatter(x=[grpmean[0]], y=[grpmean[1]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="pentagon"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="diamond-tall"))
    fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="PO-GRP_mm_ext", showlegend=True, marker=dict(size=15, symbol="diamond-wide"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="PO-GRP-cones", showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpcones_ext[0]], y=[grpcones_ext[1]], mode="markers", name="PO-GRP-cones_ext", showlegend=True, marker=dict(size=15, symbol="cross"))
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
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # fig.update_traces(marker=dict(size=15, symbol="x"))

    # lines to show the polyhedron
    # fig.add_scatter(x=[0.2, 0.45, 0.55, 0.2], y=[0.4, 0.4, 0.1, 0.4], mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    fig.add_scatter(x=create_line_path(conv_rp_arr[:, 0]), y=create_line_path(conv_rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)

    # FOR saving as pdf
    fig.update_layout(autosize=False, width=800, height=800)
    fig.write_image(f"/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/misc/{name}.pdf", width=800, height=800)

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
    grpmm = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1], "f_3": GRP[2]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1], "f_3": GRP_ext[2]}

    # find GRP, returns np.array
    grpcones = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment_optDMb(case_name):
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
    agg_rps_test_mm(problem, reference_points, cip, case_name)


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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment1_solution_process1(case_name):
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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_solution_process2(case_name):
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
    agg_rps_test_eq(problem, reference_points, cip, case_name)

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
    agg_rps_test_eq(problem, reference_points, cip, case_name)

def experiment2_test(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 3
    # ZDT3 has issues with something, maybe normalization i dunno
    # problem = zdt1(n_variables)
    problem = zdt2(n_variables)

    # eq_example_optdm1
    # verticaldms
    reference_points = [
        {"f_1": 0.9, "f_2": 0.},
        {"f_1": 0.9, "f_2": 0.11},
        {"f_1": 0.39, "f_2": 0.5},
        {"f_1": 0.0, "f_2": 0.9},
    ]
    # TODO: NOTE, LETS WRITE, LETS ASSUME AT SOME LATER ITERATION I, WE HAVE CIP AT 0.7,0.7
    cip = np.array([1, 1])

    # agg_rps_test
    agg_rps_test_eq(problem, reference_points, cip, case_name)

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
        {"f_1": 0.5, "f_2": 0.8},
        {"f_1": 0.45, "f_2": 0.5},
        {"f_1": 0.5, "f_2": 0.45},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def experiment1_zdt2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.8, "f_2": 0.4},
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_optdm1(case_name):
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.4, "f_2": 0.3},
        {"f_1": 0.8, "f_2": 0.7},
        {"f_1": 0.7, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

def exp_vertdms(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt1(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4},
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.4, "f_2": 0.8},
    ]
    cip = np.array([1, 1])

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)

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
    agg_rps_test_mm(problem, reference_points, cip, case_name)


def exp_cip2(case_name):
   # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt2(n_variables)

    # eq_example_optdm1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.4},
        {"f_1": 0.55, "f_2": 0.1},
    ]
    cip = np.array([0.8, 0.5])

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip, case_name)


if __name__ == "__main__":

    experiment_optDM2("eo3")
    # experiment_optDMb("eo2")
    # experiment_zdt3("ezdt3") # does not work

    # experiment1_solution_process1("e1s1")
    # experiment1_solution_process2("e1s2")
    # experiment1_solution_process3("e1s3")
    # experiment1_pessmistic1("e1p1")
    # experiment1_zdt2("zdt2example")
    # exp_optdm1("optdm1")
    # exp_optdm1_2("eo1")
    # exp_vertdms("verticaldms")

    # exp_change1("change1")
    # exp_change2("change2")
    # exp_change3("change3")

    # exp_shapePF("shape1")
    # exp_shapePF2("shape2")

    # exp_cip1("cip1")
    # exp_cip2("cip2")

    # EQ variants
    # experiment2_solution_process1("e2s1")
    # experiment2_solution_process2("e2s2")
    # experiment2_solution_process3("e2s3")

    # Below dtlz2 does not work for some reason.
    # experiment2_test("test")

    # experiment2_test("test")
    # experiment2_("")
