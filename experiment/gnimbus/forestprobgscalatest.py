import desdeo
import numpy as np

from desdeo.problem import (PolarsEvaluator, Problem, VariableType,
                            flatten_variable_dict, Constraint,
                            ConstraintTypeEnum, Variable, VariableTypeEnum, ScalarizationFunction
                            )
from desdeo.tools.utils import (
    get_corrected_ideal_and_nadir,
    get_corrected_reference_point,
)
from desdeo.tools.scalarization import objective_dict_has_all_symbols

from desdeo.problem.testproblems import dtlz2, nimbus_test_problem

# from desdeo.mcdm.gnimbus import infer_classifications, solve_intermediate_solutions, solve_sub_problems
from gnimbus import (explain, voting_procedure, infer_classifications, agg_cardinal, infer_ordinal_classifications,
                     solve_intermediate_solutions, solve_sub_problems, convert_to_nimbus_classification, add_group_nimbusv2_sf_diff,
                     list_of_rps_to_dict_of_rps, dict_of_rps_to_list_of_rps)

from desdeo.problem.testproblems import dtlz2, nimbus_test_problem, zdt1, zdt2, forest_problem, forest_problem_discrete
from desdeo.tools import IpoptOptions, PyomoIpoptSolver, add_asf_diff

from aggregate_classifications import aggregate_classifications

from gnimbus import *

from desdeo.tools import (
    BaseSolver,
    SolverOptions,
    SolverResults,
    add_group_asf_diff,
    add_group_asf,
    add_group_guess_sf_diff,
    add_group_guess_sf,
    add_group_nimbus_sf_diff,
    add_group_nimbus_sf,
    add_nimbus_sf_diff,
    add_nimbus_sf_nondiff,
    add_group_stom_sf_diff,
    add_group_stom_sf,
    guess_best_solver,
    add_asf_diff,
    ScalarizationError,
    add_nimbus_sf_diff, add_nimbus_sf_nondiff
)
from desdeo.mcdm.nimbus import (
    generate_starting_point,
    infer_classifications,
    NimbusError
)

from desdeo.tools.scalarization import add_asf_diff
from desdeo.tools import guess_best_solver

from desdeo.mcdm.nimbus import generate_starting_point

from aggregate_classifications import aggregate_classifications
import math
import numpy as np
from pathlib import Path
import polars as pl

from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    DiscreteRepresentation,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
from desdeo.tools.utils import available_solvers, payoff_table_method

from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    DiscreteRepresentation,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Simulator,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
def dmitry_forest_problem_discrete() -> Problem:
    """Implements the dmitry forest problem using Pareto front representation.

    Returns:
        Problem: A problem instance representing the forest problem.
    """
    filename = "/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/gnimbus/dmitry_forest_problem_non_dom_solns.csv"


    obj_names = ["Rev", "HA", "Carb", "DW"]

    var_name = "index"

    data = pl.read_csv(
        filename, has_header=True, columns=["Rev", "HA", "Carb", "DW"], separator=",", #decimal_comma=True
    )

    variables = [
        Variable(
            name=var_name,
            symbol=var_name,
            variable_type=VariableTypeEnum.integer,
            lowerbound=0,
            upperbound=len(data) - 1,
            initial_value=0,
        )
    ]

    objectives = [
        Objective(
            name=obj_name,
            symbol=obj_name,
            objective_type=ObjectiveTypeEnum.data_based,
            ideal=data[obj_name].max(),
            nadir=data[obj_name].min(),
            maximize=True,
        )
        for obj_name in obj_names
    ]

    discrete_def = DiscreteRepresentation(
        variable_values={"index": list(range(len(data)))},
        objective_values=data[[obj.symbol for obj in objectives]].to_dict(),
    )

    return Problem(
        name="Dmitry Forest Problem (Discrete)",
        description="Defines a forest problem with four objectives: revenue, habitat availability, carbon storage, and deadwood.",
        variables=variables,
        objectives=objectives,
        discrete_representation=discrete_def,
        is_twice_differentiable=False,
    )


if __name__=="__main__":
    from desdeo.problem.testproblems import forest_problem_discrete, forest_problem
    import polars as pl
    from desdeo.mcdm.nimbus import generate_starting_point

    """

    forest_problem = dtlz2(30,3)
    #ideal, nadir = payoff_table_method(problem=forest_problem)
    saved_solutions = []
    # get some initial solution
    ideal = forest_problem.get_ideal_point()
    nadir = forest_problem.get_nadir_point()
    initial_rp = ideal

    print("ideal: ", forest_problem.get_ideal_point())
    print("nadir: ", forest_problem.get_nadir_point())


    initial_result = generate_starting_point(forest_problem, initial_rp )#, initial_rp)
    print(initial_result.optimal_objectives)
    # for first iteration
    next_current_solution = initial_result.optimal_objectives
    print(f"initial solution: {next_current_solution}")

    # ITERATION 1
    reference_points = {
        "DM1": { # K
            "f_1": 0, 
            "f_2": 1, 
            "f_3": 0.5, 
        },
        "DM2": { # F
            "f_1": 1, 
            "f_2": 0, 
            "f_3": 0.3, 
        },
        "DM3": { # B
            "f_1": 1, 
            "f_2": 1, 
            "f_3": 0, 
        },
        "DM4": { # J
            "f_1": 0, 
            "f_2": 1, 
            "f_3": 1, 
        },
    }

    num_desired = 4 
    solutions = solve_sub_problems(
        forest_problem, next_current_solution, reference_points, num_desired, decision_phase=False,
    )

    ## !! TODO: TRY USING THESE PO SOLUTIONS AS THE REFERENCE POINTS FOR GROUP SCALA'S, niin sanottu Pareto skaalattuja sitten ns. fair. Estääkö nyk ongelman?
    # TODO
    nimbusdm1 = solutions[0].optimal_objectives
    nimbusdm2 = solutions[1].optimal_objectives
    nimbusdm3 = solutions[2].optimal_objectives
    nimbusdm4 = solutions[3].optimal_objectives

    # Group solutions
    gstom = solutions[4].optimal_objectives
    gasf = solutions[5].optimal_objectives
    gguess = solutions[6].optimal_objectives

    for idx, s in enumerate(solutions):
        #if idx < len(reference_points):
        #    print(f" Index {idx},Individual Solution : {s.optimal_objectives}")
        #else: 
        print(f" Index {idx}, Group Solution : {s.optimal_objectives}")


    '_alpha + 1e-06*(f_1_min / (0 - -1e-06) + f_2_min / (1 - -1e-06) + f_3_min / (0.5 - -1e-06))'

    '_alpha + 1e-06*((-1000000.0 * f_1_min) + (1.000001000001 * f_2_min) + (2.000004000008 * f_3_min) 
    
    ## + (1.000001000001 * f_1_min) + (-1000000.0 * f_2_min) + (3.3333444444814813 * f_3_min) + (1.000001000001 * f_1_min) + (1.000001000001 * f_2_min) + (-1000000.0 * f_3_min) + (-1000000.0 * f_1_min) + (1.000001000001 * f_2_min) + (1.000001000001 * f_3_min))'
    """
    
    

    forest_problem = dmitry_forest_problem_discrete()
    ideal, nadir = payoff_table_method(problem=forest_problem)
    saved_solutions = []
    # get some initial solution
    initial_rp = ideal
    
    print("ideal: ", ideal)
    print("nadir: ", nadir)

    initial_result = generate_starting_point(forest_problem, initial_rp )#, initial_rp)
    print(initial_result.optimal_objectives)

    # for first iteration
    next_current_solution = initial_result.optimal_objectives
    print(f"initial solution: {next_current_solution}")

# stom
# '(Rev_min - -249.96656415420023) / (-249.96656315420023 - -249.96656415420023), (HA_min - -20225.257708201425) / (-20225.257707201425 - -20225.257708201425), (Carb_min - -4449.001445090009) / (-3800 - -4449.001445090009), (DW_min - -218.15314456691328) / (-81 - -218.15314456691328)'

# stom -1%
# '(Rev_min - -249.96656415420023) / (-247.46689752265823 - -249.96656415420023), (HA_min - -20225.257708201425) / (-20023.00513012941 - -20225.257708201425), (Carb_min - -4449.001445090009) / (-3800 - -4449.001445090009), (DW_min - -218.15314456691328) / (-81 - -218.15314456691328)'
# ^  (-247.46689752265823 - -249.96656415420023) <- weight parth, these match the weights below as expected

    # ITERATION 1
    reference_points = {
            # gstom {'Rev': 1000000.0025247573, 'HA': 999999.6614643587, 'Carb': 0.0015408286184344499, 'DW': 0.007291119741787089}
            # gguess{'Rev': 0.005785315461423217, 'HA': 0.00012137191892471277, 'Carb': 0.001032296808501996, 'DW': 1.2503009530899023}
            
            # gstom -1% {'Rev': 0.4000533460668386, 'HA': 0.0049443127476177, 'Carb': 0.0015408286184344499, 'DW': 0.007291119741787089}
            # gguess {'Rev': 0.005870206598426078, 'HA': 0.0001244263090057934, 'Carb': 0.001032296808501996, 'DW': 1.2503009530899023}

            # gstom -0.1% {'Rev': 4.000519056879065, 'HA': 0.049443125276011336, 'Carb': 0.0015408286184344499, 'DW': 0.007291119741787089}
            # gguess {'Rev': 0.005793693927439708, 'HA': 0.00012167059326150425, 'Carb': 0.001032296808501996, 'DW': 1.2503009530899023}
        "DM1": { # Kaisa
                "Rev": ideal["Rev"], # - (0.01*ideal["Rev"]), # Revenue
                "HA": ideal["HA"], # - (0.01*ideal["HA"]), # habitat availability
                "Carb": 3800, # Carb ¤ Kaisa has <> meaning any value,  nadir["Carb"]. Moderator tried difffernet any values also.
                "DW": 81, # Deadwood
            },
            # gstom {'Rev': 0.005785315461423217, 'HA': 0.00012137191892471277, 'Carb': 0.004016040949635937, 'DW': 0.007291119741787089}
            # gguess {'Rev': 1000000.0025247573, 'HA': 999999.6614643587, 'Carb': 0.0007306130150341051, 'DW': 1.2503009530899023}
            
            # gstom +1% {'Rev': 0.005811241468446161, 'HA': 0.00012316367742515262, 'Carb': 0.004016040949635937, 'DW': 0.007291119741787089}
            # gguess {'Rev': 1.2967604880613153, 'HA': 0.00834298357871852, 'Carb': 0.0007306130150341051, 'DW': 1.2503009530899023}

            # gstom +0.1% {'Rev': 0.005787897647620006, 'HA': 0.0001215487454066376, 'Carb': 0.004016040949635937, 'DW': 0.007291119741787089}
            # gguess {'Rev': 12.967453539482063, 'HA': 0.083429829522696, 'Carb': 0.0007306130150341051, 'DW': 1.2503009530899023}
        "DM2": { # Curro
                "Rev": nadir["Rev"],# + (0.01*nadir["Rev"]) , # Revenue
                "HA": nadir["HA"],# + (0.01*nadir["HA"]), # habitat availability
                "Carb": 4200, # Carb
                "DW": 81, # Deadwood
            },
        "DM3": { # Babooshka
            "Rev": 210 , # Revenue
            "HA": 17700, # habitat availability
            "Carb": 3900, # Carb
            "DW": 185, # Deadwood
        },
        "DM4": { # Juho
            "Rev": nadir["Rev"], # Revenue
            "HA": nadir["HA"], # habitat availability
            "Carb": 3800, # Carb
            "DW": 210, # Deadwood
        },
    }



    num_desired = 4 
    solutions = solve_sub_problems(
        forest_problem, next_current_solution, reference_points, num_desired, decision_phase=False,
    )

    #nimbusdm1 = solutions[0].optimal_objectives
    #nimbusdm2 = solutions[1].optimal_objectives
    #nimbusdm3 = solutions[2].optimal_objectives
    #nimbusdm4 = solutions[3].optimal_objectives

    # Group solutions
    #gstom = solutions[4].optimal_objectives
    #gasf = solutions[5].optimal_objectives
    #gguess = solutions[6].optimal_objectives

    # stom dm1, guess dm1, stom dm2, guess dm2, gstom, gasf, gguess

    for idx, s in enumerate(solutions):
        if idx < len(reference_points):
            print(f" Index {idx},Individual Solution : {s.optimal_objectives}")
        else: 
            print(f" Index {idx}, Group Solution : {s.optimal_objectives}")

 