from desdeo.problem import *
from desdeo.tools import *
from desdeo.tools.scalarization import *
import numpy as np
# noqa

from preference_aggregation import *

from testprobs import *

import matplotlib.pyplot as plt

from pymoo.problems import get_problem
from pymoo.util.plotting import plot

import plotly.express as px
    
import pandas as pd

import plotly.io as pio   
pio.kaleido.scope.mathjax = None

"""
FOR GNIMBUS TESTS
"""




def test_add_group_asf_diff(problem: Problem, GRP: dict[str, float], prefs: list[dict[str, float]]):
    """
    Idea is to compare asf and g_asf given a group reference point and set of reference points.

    Args:
        problem (Problem): _description_
        GRP (dict[str, float]): _description_
        prefs (list[dict[str, float]]): _description_

    Returns:
        _type_: _description_
    """

    p, target = add_asf_diff(problem, "target", GRP)
    problem_w_group_sf, group_sf = add_group_asf_diff(problem, "group_sf", prefs)

    print("Testing ASF")
    # for 1 RP
    solver = PyomoIpoptSolver(p)
    res = solver.solve(target)
    xs = res.optimal_variables
    fs = res.optimal_objectives
    print(fs)

    # for group
    #create_solver = guess_best_solver(problem_w_group_sf)
    #group_solver = create_solver(problem_w_group_sf)
    group_solver = PyomoIpoptSolver(problem_w_group_sf)

    res_group = group_solver.solve(group_sf)
    gxs = res_group.optimal_variables
    gfs = res_group.optimal_objectives
    print(gfs)
        # convert for numpy
    #fs = objective_dict_to_numpy_array(problem, fs)
    #gfs = objective_dict_to_numpy_array(problem, gfs)

    return fs, gfs


def test_add_group_guess_diff(problem: Problem, GRP: dict[str, float], prefs: list[dict[str, float]]):
    """
    Idea is to compare guess and g_guess given a group reference point and set of reference points.

    Args:
        problem (Problem): _description_
        GRP (dict[str, float]): _description_
        prefs (list[dict[str, float]]): _description_

    Returns:
        _type_: _description_
    """

    p, target = add_guess_sf_diff(problem, "target", GRP)
    problem_w_group_sf, group_sf = add_group_guess_sf_diff(problem, "group_sf", prefs)

    print("Testing GUESS")
    # for 1 RP
    solver = PyomoIpoptSolver(p)
    res = solver.solve(target)
    xs = res.optimal_variables
    fs = res.optimal_objectives
    print(fs)

    # for group
    #create_solver = guess_best_solver(problem_w_group_sf)
    #group_solver = create_solver(problem_w_group_sf)
    group_solver = PyomoIpoptSolver(problem_w_group_sf)

    res_group = group_solver.solve(group_sf)
    gxs = res_group.optimal_variables
    gfs = res_group.optimal_objectives
    print(gfs)
        # convert for numpy
    #fs = objective_dict_to_numpy_array(problem, fs)
    #gfs = objective_dict_to_numpy_array(problem, gfs)

    return fs, gfs


def test_add_group_stom_diff(problem: Problem, GRP: dict[str, float], prefs: list[dict[str, float]]):
    """
    Idea is to compare stom and g_stom given a group reference point and set of reference points.

    Args:
        problem (Problem): _description_
        GRP (dict[str, float]): _description_
        prefs (list[dict[str, float]]): _description_

    Returns:
        _type_: _description_
    """

    p, target = add_stom_sf_diff(problem, "target", GRP)
    problem_w_group_sf, group_sf = add_group_stom_sf_diff(problem, "group_sf", prefs)

    # for 1 RP
    solver = PyomoIpoptSolver(p)
    res = solver.solve(target)
    xs = res.optimal_variables
    fs = res.optimal_objectives
    print(fs)

    # for group
    #create_solver = guess_best_solver(problem_w_group_sf)
    #group_solver = create_solver(problem_w_group_sf)
    group_solver = PyomoIpoptSolver(problem_w_group_sf)

    res_group = group_solver.solve(group_sf)
    gxs = res_group.optimal_variables
    gfs = res_group.optimal_objectives
    print(gfs)
   
    # convert for numpy
    #fs = objective_dict_to_numpy_array(problem, fs)
    #gfs = objective_dict_to_numpy_array(problem, gfs)

    return fs, gfs


def test_asf_individual_sols(problem: Problem, testname, rps: list[dict[str, float]] , pa: str):
    """
        Solves using asf first the individual Pareto optimal solutions for each DM. 
        Then finds GRP and projected GRP according to the preference aggregation method selected.
        Then, solves asf and g_asf by using GRP and all RPs respectively.
        Then, solves asf and g_asf using projected GRP and individual Pareto optimal solutions for each DM.
        Finally, calls to plot all these for comparisons.

    """

    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # for maxmins
    cip = nadir
    #cip = {"f_1": np.max(rps["f_1"]), "f_2": np.max(rps["f_2"])}
    k = len(ideal)
    q = len(rps)

    all_prefs = rps # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    print(cip)

    ind_sols = []
    for rp in range(len(rp_arr)):
        # get scala and solve for all
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, rp_arr[rp]) 
        p, target = add_asf_diff(problem, f"target{rp}", dm_rp)

        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs = res.optimal_objectives
        #print(fs)
        ind_sols.append(fs)

    # TODO: give ind_sols to group scalas. Can use existing stuff, but do not remove the functionality of GRP so just repeat and comment.

    # find GRP for preferences, returns np.array
    grp2 = find_GRP(rp_arr, cip, k, q, ideal, nadir, pa)

    ind_sols_arr = np.array([[col["f_1"], col["f_2"]] for col in ind_sols])
    # find GRP for projected preferences, returns np.array
    proj_grp = find_GRP(ind_sols_arr, cip, k, q, ideal, nadir, pa)

    # make dict from the GPR array
    GRP = {"f_1": grp2[0], "f_2": grp2[1]}
    proj_GRP = {"f_1": proj_grp[0], "f_2": proj_grp[1]}

    asf, group_asf_sol = test_add_group_asf_diff(problem, GRP, all_prefs) #just ignore GRP for now..s
    asf2, group_asf_sol2 = test_add_group_asf_diff(problem, proj_GRP, ind_sols)
    #guess, group_guess_sol = test_add_group_guess_diff(problem, GRP, all_prefs)
    #stom, group_stom_sol = test_add_group_stom_diff(problem, GRP, all_prefs)

    solutions = [asf, group_asf_sol, asf2, group_asf_sol2]
    keys = ["f_1", "f_2"]
    namelist = ["GRP_asf", "g_asf", "GRP_ind_asf", "ind_g_asf"]
    for i in range(len(all_prefs)):
        solutions.append(ind_sols[i])
        namelist.append(f"indsol{i}")
   
    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }
   
    all_solutions = pd.DataFrame(all_solutions, columns= ["f_1", "f_2", "names"])
    visualize_2d(problem, testname, rp_arr, grp2, all_solutions, keys, namelist,  proj_grp)


def test_stom_individual_sols(problem: Problem, testname, rps: list[dict[str, float]] , pa: str):
    """
        Solves using stom first the individual Pareto optimal solutions for each DM. 
        Then finds GRP and projected GRP according to the preference aggregation method selected.
        Then, solves stom and g_stom by using GRP and all RPs respectively.
        Then, solves stom and g_stom using projected GRP and individual Pareto optimal solutions for each DM.
        Finally, calls to plot all these for comparisons.

    """

    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # for maxmins
    cip = nadir
    #cip = {"f_1": np.max(rps["f_1"]), "f_2": np.max(rps["f_2"])}
    k = len(ideal)
    q = len(rps)

    all_prefs = rps # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    print(cip)

    ind_sols = []
    for rp in range(len(rp_arr)):
        # get scala and solve for all
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, rp_arr[rp]) 
        p, target = add_stom_sf_diff(problem, f"target{rp}", dm_rp)

        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs = res.optimal_objectives
        #print(fs)
        ind_sols.append(fs)

    # TODO: give ind_sols to group scalas. Can use existing stuff, but do not remove the functionality of GRP so just repeat and comment.

    # find GRP, returns np.array
    grp2 = find_GRP(rp_arr, cip, k, q, ideal, nadir, pa)

    ind_sols_arr = np.array([[col["f_1"], col["f_2"]] for col in ind_sols])
    # find GRP for projected preferences, returns np.array
    proj_grp = find_GRP(ind_sols_arr, cip, k, q, ideal, nadir, pa)

    # make dict from the GPR array
    GRP = {"f_1": grp2[0], "f_2": grp2[1]}
    proj_GRP = {"f_1": proj_grp[0], "f_2": proj_grp[1]}

    asf, group_asf_sol = test_add_group_stom_diff(problem, GRP, all_prefs) #just ignore GRP for now..s
    asf2, group_asf_sol2 = test_add_group_stom_diff(problem, proj_GRP, ind_sols)
    #guess, group_guess_sol = test_add_group_guess_diff(problem, GRP, all_prefs)
    #stom, group_stom_sol = test_add_group_stom_diff(problem, GRP, all_prefs)

    solutions = [asf, group_asf_sol, asf2, group_asf_sol2]
    keys = ["f_1", "f_2"]
    namelist = ["GRP_stom", "g_stom", "GRP_ind_stom", "ind_g_stom"]
    for i in range(len(all_prefs)):
        solutions.append(ind_sols[i])
        namelist.append(f"indsol{i}")
   
    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }
   
    all_solutions = pd.DataFrame(all_solutions, columns= ["f_1", "f_2", "names"])
    visualize_2d(problem, testname, rp_arr, grp2, all_solutions, keys, namelist, proj_grp)

def test_guess_individual_sols(problem: Problem, testname, rps: list[dict[str, float]] , pa: str):
    """
        Solves using guess first the individual Pareto optimal solutions for each DM. 
        Then finds GRP and projected GRP according to the preference aggregation method selected.
        Then, solves guess and g_guess by using GRP and all RPs respectively.
        Then, solves guess and g_guess using projected GRP and individual Pareto optimal solutions for each DM respectively.
        Finally, calls to plot all these for comparisons.

    """
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # for maxmins
    cip = nadir
    #cip = {"f_1": np.max(rps["f_1"]), "f_2": np.max(rps["f_2"])}
    k = len(ideal)
    q = len(rps)

    all_prefs = rps # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    print(cip)

    ind_sols = []
    for rp in range(len(rp_arr)):
        # get scala and solve for all
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, rp_arr[rp]) 
        p, target = add_guess_sf_diff(problem, f"target{rp}", dm_rp)

        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        xs = res.optimal_variables
        fs = res.optimal_objectives
        #print(fs)
        ind_sols.append(fs)

    # TODO: give ind_sols to group scalas. Can use existing stuff, but do not remove the functionality of GRP so just repeat and comment.

    # find GRP, returns np.array
    grp2 = find_GRP(rp_arr, cip, k, q, ideal, nadir, pa)
    ind_sols_arr = np.array([[col["f_1"], col["f_2"]] for col in ind_sols])
    # find GRP for projected preferences, returns np.array
    proj_grp = find_GRP(ind_sols_arr, cip, k, q, ideal, nadir, pa)
    # make dict from the GPR array
    GRP = {"f_1": grp2[0], "f_2": grp2[1]}
    proj_GRP = {"f_1": proj_grp[0], "f_2": proj_grp[1]}

    asf, group_asf_sol = test_add_group_guess_diff(problem, GRP, all_prefs) #just ignore GRP for now..s
    asf2, group_asf_sol2 = test_add_group_guess_diff(problem, proj_GRP, ind_sols)
    #guess, group_guess_sol = test_add_group_guess_diff(problem, GRP, all_prefs)
    #stom, group_stom_sol = test_add_group_stom_diff(problem, GRP, all_prefs)

    solutions = [asf, group_asf_sol, asf2, group_asf_sol2]
    keys = ["f_1", "f_2"]
    namelist = ["GRP_guess", "g_guess", "GRP_ind_guess", "ind_g_guess"]
    for i in range(len(all_prefs)):
        solutions.append(ind_sols[i])
        namelist.append(f"indsol{i}")
   
    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }
   
    all_solutions = pd.DataFrame(all_solutions, columns= ["f_1", "f_2", "names"])

    visualize_2d(problem, testname,rp_arr, grp2, all_solutions, keys, namelist, proj_grp)


def test_group_scalas_and_maxmin(problem: Problem, testname, rps: list[dict[str, float]] , pa: str):
    """
        Solves using the scalarization functions using GRP and group scalarization functions using all preferences.

    """
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # for maxmins
    cip = nadir
    #cip = {"f_1": np.max(rps["f_1"]), "f_2": np.max(rps["f_2"])}
    k = len(ideal)
    q = len(rps)

    all_prefs = rps # !! is assumed to be list or smh in test_add_group_asf_diff jne
    # convert all_prefs to numpy array for find_GRP or find GRP should handle dictionaries.. which makes more sense TODO
    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])
    #cip = np.array(np.amax(rp_arr, axis=0))
    #print(cip)

    # find GRP, returns np.array
    grp2 = find_GRP(rp_arr, cip, k, q, ideal, nadir, pa)
    # make dict from the GPR array
    GRP = {"f_1": grp2[0], "f_2": grp2[1]}

    # solve with each scalarization function (with GRP) and group scalarization function (with all prefs).
    asf, group_asf_sol = test_add_group_asf_diff(problem, GRP, all_prefs)
    guess, group_guess_sol = test_add_group_guess_diff(problem, GRP, all_prefs)
    stom, group_stom_sol = test_add_group_stom_diff(problem, GRP, all_prefs)

    # move all generated solutions to a a dataframe for plotting
    solutions = [asf, group_asf_sol, guess, group_guess_sol, stom, group_stom_sol]
    keys = ["f_1", "f_2"]
    namelist = ["GRP_asf", "g_asf", "GRP_guess", "g_guess", "GRP_stom", "g_stom"]
    all_solutions = {
        "f_1": [s["f_1"] for s in solutions],
        "f_2": [s["f_2"] for s in solutions],
        "names": namelist,
    }
    all_solutions = pd.DataFrame(all_solutions, columns= ["f_1", "f_2", "names"])

    # TODO: figure out the marker styles better
    visualize_2d(problem, testname, rp_arr, grp2, all_solutions, keys, namelist, None)


def visualize_2d(problem, testname, rp_arr, grp2, all_solutions, keys, namelist, proj_grp: None):
    # plot PF
    problem = get_problem(problem.name)
    pf = problem.pareto_front()
    PF = [dict(zip(keys, row)) for row in pf]   
    fig = px.line(PF, x="f_1", y="f_2")

    for i in range(len(rp_arr)):
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP",showlegend=True, marker=dict(size=15, symbol="x"))
    # PLOT GRPs
    fig.add_scatter(x=[grp2[0]], y=[grp2[1]], mode="markers", name="GRP",showlegend=True, marker=dict(size=20, symbol="diamond-wide"))
    if proj_grp is not None:
        fig.add_scatter(x=[proj_grp[0]], y=[proj_grp[1]], mode="markers", name="proj_GRP",showlegend=True, marker=dict(size=20, symbol="diamond-tall")) 
    # plot all solutions
    for i in range(len(namelist)):
        fig.add_scatter(x=[all_solutions["f_1"][i]], y=[all_solutions["f_2"][i]], mode="markers", name=all_solutions.names[i] ,showlegend=True, marker=dict(size=15))
    # create valid area path for GRP
    fig.add_scatter(x=create_line_path(rp_arr[:,0]), y=create_line_path(rp_arr[:, 1]), mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    #fig.update_traces(marker=dict(size=15))

    fig.write_image(f"/home/jp/tyot/mop/papers/GNIMBUS/graphs/gscalatests/{testname}.pdf", width=800, height=800)
    fig.write_html(f"/home/jp/tyot/mop/papers/GNIMBUS/graphs/gscalatests/{testname}.html")

    fig.show()  

def create_line_path(arr):
    return np.append(arr, arr[0])


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

    rp = {"f_1": 0.5, "f_2": 0.6 }
    #rp1 = {"f_1": 0.9, "f_2": 0.6, }
    #rp2 = {"f_1": 0.55, "f_2": 0.6, }
    #rp3 = {"f_1": 0.0, "f_2": 0.1, }
    rp1 = {"f_1": 0.6, "f_2": 0.30 }
    rp2 = {"f_1": 0.34, "f_2": 0.32 }
    rp3 = {"f_1": 0.15, "f_2": 0.0}
    # Data for plotting
    rp_np =  [0.5, 0.6]

    rp1_np = [0.6, 0.3]
    rp2_np = [0.34, 0.32]
    rp3_np = [0.15, 0.0]

    cip = np.array([1.,1.])
    nadir = np.array([1.,1.])
    ideal = np.array([0.,0.])
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
         asf_sol, group_asf_sol, guess_sol, group_guess_sol,stom_sol, group_stom_sol
    ]
    problem = get_problem("zdt1")
    pf = problem.pareto_front()

    colors = ["green", "red", "b", "orange", "m", "purple", "k"]

    fig, ax = plt.subplots()
    ax.plot(pf[:,0], pf[:,1])

    # plot RPs
    #ax.scatter(rp_np[0], rp_np[1], c=colors[-1], label="RP", marker="s")
    ax.scatter(rp1_np[0], rp1_np[1], c=colors[0], label="DM1_RP", marker="s")
    ax.scatter(rp2_np[0], rp2_np[1], c=colors[1], label="DM2_RP", marker="s")
    ax.scatter(rp3_np[0], rp3_np[1], c=colors[2], label="DM3_RP", marker="s")
    ax.scatter(grp2[0], grp2[1], c=colors[0], label="MAXMINGRP", marker="X")
 
    ax.scatter(asf_sol[0], asf_sol[1], c=colors[0], label="asf", marker="x")
    ax.scatter(group_asf_sol[0], group_asf_sol[1], c=colors[1], label="gasf", marker="x")

    ax.scatter(guess_sol[0], guess_sol[1], c=colors[2], label="guess",  marker="P")
    ax.scatter(group_guess_sol[0], group_guess_sol[1], c=colors[3], label="g_guess",  marker="P")

    ax.scatter(stom_sol[0], stom_sol[1], c=colors[4], label="stom",  marker="*")
    ax.scatter(group_stom_sol[0], group_stom_sol[1], c=colors[5], label="g_stom",  marker="*")

    #fig.add_scatter(x=create_line_path(rp_arr[:,0]), y=create_line_path(rp_arr[:, 1]), mode="lines", line=dict(color="#808080"), name="valid_area", showlegend=True)
    

    ax.set(xlabel='f_1', ylabel='f_2',title='Group scalarization maxmin')
    ax.grid()
    ax.legend()

    #fig.savefig("gscalares/maxmin.png")
    plt.show()



if __name__ == "__main__":

    # pa = "maxmin" # maxmin brings the same result as the group scalarization functions because the 
    # formulations are very similar. TEST if this is really the case always!!
    # CONCLUISON: NOT TRUE, only sometimes.
    #test_maxmin()
    n_variables = 30
    n_objectives = 2
    problem = zdt1(n_variables)

    # eq_example_optdm1 TODO: update these
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4}, 
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.6, "f_2": 0.45},
        {"f_1": 0.8, "f_2": 0.25},
        {"f_1": 0.5, "f_2": 0.2},
    ]
    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1., 1.])
    #cip = np.array([0.8, 0.5])
    pa = "maxmin"

    test_group_scalas_and_maxmin(problem, 
        "e2maxmin",
        reference_points, 
        pa)
    
    test_asf_individual_sols(problem, "e2_asf", reference_points, pa)
    test_stom_individual_sols(problem, "e2_stom", reference_points, pa)
    test_guess_individual_sols(problem,"e2_guess", reference_points, pa)


    """
        # dq1 test_group_scalas_and_maxmin all the same PO sols
    reference_points = [
        {"f_1": 0.2, "f_2": 0.4}, 
        {"f_1": 0.1, "f_2": 0.8},
        {"f_1": 0.4, "f_2": 0.45},
        {"f_1": 0.8, "f_2": 0.1},
        {"f_1": 0.4, "f_2": 0.2},
    ]
    # zdt3 gets local optimal for here if moving cip from nadir sometimes
    cip = np.array([1., 1.])
    #cip = np.array([0.8, 0.5])
    pa = "maxmin_cones"
    """