
from gnimbus import *


if __name__ == "__main__":
    from desdeo.problem.testproblems import forest_problem_discrete
    import polars as pl

    # TODO: update this
    forest_problem = forest_problem_discrete()  # or use forest_problem()
    total_steps = 5

    for i in range(3):
        print(forest_problem.objectives[i].ideal)
        print(forest_problem.objectives[i].nadir)

    # get some initial solution
    initial_rp = {
        "f_1": 2000, "f_2": 50000, "f_3": 75000
    }
    problem_w_sf, target = add_asf_diff(forest_problem, "target", initial_rp)
    solver = guess_best_solver(problem_w_sf)
    initial_result = solver.solve(target)

    next_current_solution = initial_result.optimal_objectives

    # ITERATION LOOP FOR LEARNING PHASE
    reference_points = {
        "DM1": {
            "stock": 1000,
            "harvest_value": 1000,
            "npv": 80000,
        },
        "DM2": {
            "stock": 3000,
            "harvest_value": 22200.5,
            "npv": 72000,
        },
        "DM3": {
            "stock": 2000,
            "harvest_value": 50005,
            "npv": 80000,
        },
        "DM4": {
            "stock": 1500,
            "harvest_value": 100,
            "npv": 90000,
        },
        "DM5": {
            "stock": 3200,
            "harvest_value": 900,
            "npv": 75000,
        },
    }
    # TODO: learning phase
    num_desired = 5  # TODO: add more scalarization to get atleast 5 solutions.
    solutions = solve_sub_problems(
        problem, next_current_solution, reference_points, num_desired, decision_phase=False,
    )
    for i in solutions:
        print(solutions[i].optimal_objectives)

    votes_idx = {
        "DM1": 1,
        "DM1": 0,
        "DM1": 3,
        "DM1": 2,
        "DM1": 3,
    }

    voted_sol = voting_procedure(votes_idx)

    next_current_solution = voted_sol
    print(next_current_solution)

   # ITERATION LOOP FOR DECISION PHASE
    reference_points = {
        "DM1": {
            "stock": 1000,
            "harvest_value": 1000,
            "npv": 80000,
        },
        "DM2": {
            "stock": 3000,
            "harvest_value": 22200.5,
            "npv": 72000,
        },
        "DM3": {
            "stock": 2000,
            "harvest_value": 50005,
            "npv": 80000,
        },
        "DM4": {
            "stock": 1500,
            "harvest_value": 100,
            "npv": 90000,
        },
        "DM5": {
            "stock": 3200,
            "harvest_value": 900,
            "npv": 75000,
        },
    }
    # TODO: learning phase
    num_desired = 1
    solutions = solve_sub_problems(
        problem, next_current_solution, reference_points, num_desired, decision_phase=True,
    )
    for i in solutions:
        print(solutions[i].optimal_objectives)

    votes_idx = {
        "DM1": 1,
        "DM1": 0,
        "DM1": 3,
        "DM1": 2,
        "DM1": 3,
    }

    voted_sol = voting_procedure(votes_idx)

    next_current_solution = voted_sol
    print(next_current_solution)
