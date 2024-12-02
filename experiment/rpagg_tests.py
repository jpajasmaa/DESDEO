from desdeo.problem import Problem, objective_dict_to_numpy_array, numpy_array_to_objective_dict, get_ideal_dict, get_nadir_dict
from desdeo.tools.scalarization import add_asf_diff, add_guess_sf_diff, add_stom_sf_diff, add_asf_generic_diff
import numpy as np

from desdeo.problem.testproblems import zdt1, zdt2, zdt3
# noqa

from desdeo.tools.utils import guess_best_solver, PyomoIpoptSolver, NevergradGenericSolver

from preference_aggregation import find_GRP, find_GRP2


import matplotlib.pyplot as plt

from pymoo.problems import get_problem
from pymoo.util.plotting import plot

import plotly.express as px

import pandas as pd

import plotly.io as pio
pio.kaleido.scope.mathjax = None

"""
FOR AGGREGATING RPS TESTS
"""

"""
        TODO: clean this mess, and make real functions to do all these things
        TODO: Tests
        TODO: nice plotting
        This simple idea of using the maxmin criteria to find the "fair" group reference point. Which is how to aggregate the RPs of the DMs.
        Then, use that GRP as normal RP in scalarization functions. the function find_GRP needs nadir as the "current iteration point"
"""
def test_maxmin():
    # in nimbus, the RPs should be near PF (most cases)

    n_variables = 30
    n_objectives = 2
    problem = zdt1(n_variables)

    rp = {"f_1": 0.5, "f_2": 0.6}
    # rp1 = {"f_1": 0.9, "f_2": 0.6, }
    # rp2 = {"f_1": 0.55, "f_2": 0.6, }
    # rp3 = {"f_1": 0.0, "f_2": 0.1, }
    rp1 = {"f_1": 0.6, "f_2": 0.30}
    rp2 = {"f_1": 0.34, "f_2": 0.32}
    rp3 = {"f_1": 0.15, "f_2": 0.0}
    # Data for plotting
    rp_np = [0.5, 0.6]

    rp1_np = [0.6, 0.3]
    rp2_np = [0.34, 0.32]
    rp3_np = [0.15, 0.0]

    cip = np.array([1., 1.])
    nadir = np.array([1., 1.])
    ideal = np.array([0., 0.])
    k = 2
    q = 3
    # pa = "maxmin" # maxmin brings the same result as the group scalarization functions because the formulations are very similar. TEST if this is really the case always!!
    pa = "maxmin"

    grp = [rp1, rp2, rp3]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    grp2 = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("GRP: ", grp)
    print("MAXMIN GRP", grp2)
    GRP = {"f_1": grp2[0], "f_2": grp2[1]}

    # asf_sols etc will get overriden who cares
    asf_sol2, group_asf_sol = test_add_group_asf_diff(rp, grp)
    guess_sol2, group_guess_sol = test_add_group_guess_diff(rp, grp)
    stom_sol2, group_stom_sol = test_add_group_stom_diff(rp, grp)

    asf, asf_target = add_asf_diff(problem, symbol="target", reference_point=GRP)
    guess, guess_target = add_guess_sf_diff(problem, symbol="target", reference_point=GRP)
    stom, stom_target = add_stom_sf_diff(problem, symbol="target", reference_point=GRP)

    print("Testing ASF")
    # for 1 RP
    solver_asf = PyomoIpoptSolver(asf)
    asf_sol = solver_asf.solve(asf_target)

    solver_guess = PyomoIpoptSolver(guess)
    guess_sol = solver_guess.solve(guess_target)

    solver_stom = PyomoIpoptSolver(stom)
    stom_sol = solver_stom.solve(stom_target)

    # convert to numpy arrays for plotting
    asf_sol = objective_dict_to_numpy_array(problem, asf_sol.optimal_objectives)
    guess_sol = objective_dict_to_numpy_array(problem, guess_sol.optimal_objectives)
    stom_sol = objective_dict_to_numpy_array(problem, stom_sol.optimal_objectives)
    print("Maxmin sols:")
    print(asf_sol)
    print(guess_sol)
    print(stom_sol)

    # testing MAXMIN GRP asf, guess, stom

    all_points = [
        asf_sol, group_asf_sol, guess_sol, group_guess_sol, stom_sol, group_stom_sol
    ]
    problem = get_problem("zdt1")
    pf = problem.pareto_front()

    colors = ["green", "red", "b", "orange", "m", "purple", "k"]

    fig, ax = plt.subplots()
    ax.plot(pf[:, 0], pf[:, 1])

    # plot RPs
    # ax.scatter(rp_np[0], rp_np[1], c=colors[-1], label="RP", marker="s")
    ax.scatter(rp1_np[0], rp1_np[1], c=colors[0], label="DM1_RP", marker="s")
    ax.scatter(rp2_np[0], rp2_np[1], c=colors[1], label="DM2_RP", marker="s")
    ax.scatter(rp3_np[0], rp3_np[1], c=colors[2], label="DM3_RP", marker="s")
    ax.scatter(grp2[0], grp2[1], c=colors[0], label="MAXMINGRP", marker="X")

    ax.scatter(asf_sol[0], asf_sol[1], c=colors[0], label="asf", marker="x")
    ax.scatter(group_asf_sol[0], group_asf_sol[1], c=colors[1], label="gasf", marker="x")

    ax.scatter(guess_sol[0], guess_sol[1], c=colors[2], label="guess", marker="P")
    ax.scatter(group_guess_sol[0], group_guess_sol[1], c=colors[3], label="g_guess", marker="P")

    ax.scatter(stom_sol[0], stom_sol[1], c=colors[4], label="stom", marker="*")
    ax.scatter(group_stom_sol[0], group_stom_sol[1], c=colors[5], label="g_stom", marker="*")

    ax.set(xlabel='f_1', ylabel='f_2', title='Group scalarization maxmin')
    ax.grid()
    ax.legend()

    # fig.savefig("gscalares/maxmin.png")
    plt.show()

# TODO: only works iwth maxmin ja maxmin-cones. MAke another with eq variants
def agg_rps_test_mm(problem: Problem, rps: list[dict[str, float]], cip, solver=None, solver_options=None):

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
    grpmm = find_GRP2(rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP_ext, returns np.array
    grpmm_ext = find_GRP2(rp_arr, cip, k, q, ideal, nadir, "maxmin_ext")
    # improvmenet direction
    GRP_ext = cip - grpmm_ext
    # make dict from the GPR array
    GRP_ext = {"f_1": GRP_ext[0], "f_2": GRP_ext[1]}

    # find GRP, returns np.array
    grpcones = find_GRP2(rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

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

    """
    VISUALIZING
    """

    solutions = [fs_mean, fs_mm, fs_mm_ext, fs_cones]

    keys = ["f_1", "f_2"]
    namelist = ["PO_mean", "PO_mm", "PO_mm_ext", "PO_cones"]

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
    fig.add_scatter(x=[meanGRP_arr[0]], y=[meanGRP_arr[1]], mode="markers", name="GRP_mean", showlegend=True, marker=dict(size=15, symbol="diamond"))
    fig.add_scatter(x=[grpmm_ext[0]], y=[grpmm_ext[1]], mode="markers", name="GRP_mm_ext", showlegend=True, marker=dict(size=15, symbol="cross"))
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="GRP_mm", showlegend=True, marker=dict(size=15, symbol="diamond-tall"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="GRP-cones", showlegend=True, marker=dict(size=15, symbol="diamond-wide"))
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

    # fig.write_image("/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/misc/zdt3e1.pdf", width=600, height=600)

    fig.show()


def agg_rps_test_eq(problem: Problem, rps: list[dict[str, float]], cip):

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

    # find GRP, returns np.array
    grpmm = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin")
    # improvmenet direction
    GRP = cip - grpmm
    # make dict from the GPR array
    GRP = {"f_1": GRP[0], "f_2": GRP[1]}

    # find GRP, returns np.array
    grpcones = find_GRP(conv_rp_arr, cip, k, q, ideal, nadir, "maxmin_cones")
    # improvmenet direction
    GRP_cones = cip - grpcones
    # make dict from the GPR array
    GRP_cones = {"f_1": GRP_cones[0], "f_2": GRP_cones[1]}

    """
    SOLVING
    """

    # TODO: MEAN

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

    """
    VISUALIZING
    """

    solutions = [fs_mm, fs_cones]

    keys = ["f_1", "f_2"]
    namelist = ["PO_mm", "PO_cones"]

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
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP", showlegend=True, marker=dict(size=10, symbol="x"))
        fig.add_scatter(x=[conv_rp_arr[i][0]], y=[conv_rp_arr[i][1]], mode="markers", name=f"converted DM{
                        i+1}_RP", showlegend=True, marker=dict(size=15, symbol="hexagram"))

    # PLOT GRP
    fig.add_scatter(x=[grpmm[0]], y=[grpmm[1]], mode="markers", name="PO-GRP_mm", showlegend=True, marker=dict(size=15, symbol="diamond-tall"))
    fig.add_scatter(x=[grpcones[0]], y=[grpcones[1]], mode="markers", name="PO-GRP-cones", showlegend=True, marker=dict(size=15, symbol="diamond-wide"))
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
    fig.add_scatter(x=create_line_path(conv_rp_arr[:, 0]), y=create_line_path(conv_rp_arr[:, 1]),
                    mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)

    fig.write_image("/home/jp/tyot/mop/papers/prefagg_concept/experiment_pics/misc/eq_moved_cip.pdf", width=600, height=600)

    fig.show()

def create_line_path(arr):
    return np.append(arr, arr[0])


if __name__ == "__main__":

    # pa = "maxmin" # maxmin brings the same result as the group scalarization functions because the
    # formulations are very similar. TEST if this is really the case always!!
    # CONCLUISON: NOT TRUE, only sometimes.
    # test_maxmin()
    n_variables = 30
    n_objectives = 2
    # ZDT3 has issues with something, maybe normalization i dunno
    problem = zdt3(n_variables)

    # eq_example_optdm1
    # zdt2 dms
    reference_points = [
        {"f_1": 0.8, "f_2": 0.4},
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": -0.1},
    ]
    cip = np.array([1., 1.])
    # cip = np.array([0.8, 0.5])
    pa = "maxmin"

    # test_group_scalas(problem, reference_points, pa)

    # agg_rps_test
    agg_rps_test_mm(problem, reference_points, cip)

    # done
    # agg_rps_test_eq(problem, reference_points, cip)

    """
        # test1
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4},
        {"f_1": 0.45, "f_2": 0.5},
        {"f_1": 0.55, "f_2": 0.1},
    ]

    # test 1 cip
        reference_points = [
        {"f_1": 0.2, "f_2": 0.4}, 
        {"f_1": 0.45, "f_2": 0.4},
        {"f_1": 0.55, "f_2": 0.1},
    ]


        # change 1
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4}, 
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.2},
    ]
    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1, 1])

    # change 2
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4}, 
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.6},
    ]

    # change 3
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4}, 
        {"f_1": 0.5, "f_2": 0.6},
        {"f_1": 0.9, "f_2": 0.7},
    ]
        # verticaldms
    reference_points = [
        {"f_1": 0.3, "f_2": 0.4}, 
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.4, "f_2": 0.8},
    ]

    # zdt2 dms
    reference_points = [
        {"f_1": 0.8, "f_2": 0.4}, 
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": 0.8},
    ]


    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1, 1])



        reference_points = [
        {"f_1": 0.8, "f_2": 0.4}, 
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": 0.8},
    ]
    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1, 1])


        reference_points = [
        {"f_1": 0.8, "f_2": 0.4}, 
        {"f_1": 0.3, "f_2": 0.6},
        {"f_1": 0.5, "f_2": 0.8},
    ]
    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1, 1])
    """
