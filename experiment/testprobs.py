import numpy as np
from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    DiscreteRepresentation,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Variable,
    VariableTypeEnum,
)

def dtlz1(n_variables: int, n_objectives: int) -> Problem:
    r"""Defines the DTLZ2 test problem. TODO: update doc string

    The objective functions for DTLZ2 are defined as follows, for $i = 1$ to $M$:

    \begin{equation}
        \underset{\mathbf{x}}{\operatorname{min}}
        f_i(\mathbf{x}) = (1+g(\mathbf{x}_M)) \prod_{j=1}^{M-i} \cos\left(x_j \frac{\pi}{2}\right) \times
        \begin{cases}
        1 & \text{if } i=1 \\
        \sin\left(x_{(M-i+1)}\frac{\pi}{2}\right) & \text{otherwise},
        \end{cases}
    \end{equation}

    where

    \begin{equation}
    g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M} \left( x_i - 0.5 \right)^2,
    \end{equation}

    and $\mathbf{x}_M$ represents the last $n-k$ dimensions of the decision vector.
    Pareto optimal solutions to the DTLZ2 problem consist of $x_i = 0.5$ for
    all $x_i \in\mathbf{x}_{M}$, and $\sum{i=1}^{M} f_i^2 = 1$.

    Args:
        n_variables (int): number of variables.
        n_objectives (int): number of objective functions.

    Returns:
        Problem: an instance of the DTLZ4 problem with `n_variables` variables and `n_objectives` objective
            functions.

    References:
        Deb, K., Thiele, L., Laumanns, M., Zitzler, E. (2005). Scalable Test
            Problems for Evolutionary Multiobjective Optimization. In: Abraham, A.,
            Jain, L., Goldberg, R. (eds) Evolutionary Multiobjective Optimization.
            Advanced Information and Knowledge Processing. Springer.
    """
    #alpha = 100 # this parameter differents this from dtlz2: does not work very well for some reason.. only finding extreme solutions.
    # function g
    g_symbol = "g"
    g_expr = " + ".join([f"(x_{i} - 0.5)**2 - Cos(20*{np.pi}*(x_{i} - 0.5)) "for i in range(1, n_objectives + 1) ])
    g_expr = f"100 + {n_objectives}" + g_expr
    # return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5)), axis=1))
    #g_expr = " + ".join([f"(x_{i} - 0.5)**2" for i in range(n_objectives, n_variables + 1)])

    objectives = []
    for m in range(1, n_objectives + 1):
        # function f_m
        # how to do this..
        prod_expr = " + ".join([f"(0.5 * (x_{i}))" for i in range(1, n_objectives - m + 1)])
        if m > 1:
            prod_expr += f"{' * ' if prod_expr != "" else ""} (1 + {g_expr}) "
            #prod_expr += f"{' * ' if prod_expr != "" else ""}Sin(0.5 * {np.pi} * (x_{n_objectives - m + 1}**{alpha}))" # needed here or not?
        if prod_expr == "":
            prod_expr = "1"  # When m == n_objectives, the product is empty, implying f_M = g.
        f_m_expr = f"({g_symbol}) * ({prod_expr})"

        objectives.append(
            Objective(
                name=f"f_{m}",
                symbol=f"f_{m}",
                func=f_m_expr,
                maximize=False,
                ideal=0,
                nadir=2,  # here nadir 2 kinda breaks it.
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=True,
            )
        )

    variables = [
        Variable(
            name=f"x_{i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.real,
            lowerbound=0,
            upperbound=1,
            initial_value=1.0,
        )
        for i in range(1, n_variables + 1)
    ]

    extras = [
        ExtraFunction(
            name="g", symbol=g_symbol, func=g_expr, is_convex=False, is_linear=False, is_twice_differentiable=True
        ),
    ]

    return Problem(
        name="dtlz1",
        description="The DTLZ1 test problem.",
        variables=variables,
        objectives=objectives,
        extra_funcs=extras,
    )


def dtlz7(n_variables: int, n_objectives: int) -> Problem:

    # function g
    g_symbol = "g"
    g_expr = " + ".join([f"(x_{i}" for i in range(n_objectives, n_variables + 1)]) # or * ? how to make the sum
    g_expr = "1 + (9/8) * " + g_expr

    h_symbol = "h"
    h_expr = "-".join([f"(x_{i}/(1+{g_expr}) * (1 + Sin (3 * {np.pi} * x_{i}))) " for i in range(1, n_objectives)])
    h_expr = "3 + " + h_expr

    objectives = []
    for m in range(1, n_objectives + 1):
        # function f_m
        prod_expr = "*".join(f"x_{m}")
        # prod_expr = " * ".join([f"Cos(0.5 * {np.pi} * (x_{i}**{alpha}))" for i in range(1, n_objectives - m + 1)])
        if m == 3:
            prod_expr = f"(1 + {g_expr}) * {h_expr}"
            f_m_expr = f"({g_symbol}) * ({h_symbol})"
        f_m_expr = f"({prod_expr})"

        objectives.append(
            Objective(
                name=f"f_{m}",
                symbol=f"f_{m}",
                func=f_m_expr,
                maximize=False,
                ideal=0,
                nadir=1,  # here nadir 2 kinda breaks it.
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=True,
            )
        )

    variables = [
        Variable(
            name=f"x_{i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.real,
            lowerbound=0,
            upperbound=1,
            initial_value=1.0,
        )
        for i in range(1, n_variables + 1)
    ]

    extras = [
        ExtraFunction(
            name="g", symbol=g_symbol, func=g_expr, is_convex=False, is_linear=False, is_twice_differentiable=True
        ),
        ExtraFunction(
            name="h", symbol=h_symbol, func=h_expr, is_convex=False, is_linear=False, is_twice_differentiable=True
        ),
    ]

    return Problem(
        name="dtlz7",
        description="The DTLZ7 test problem.",
        variables=variables,
        objectives=objectives,
        extra_funcs=extras,
    )


    # dltz6 g_expr 
    #g_expr = " + ".join([f"(x_{i})**0.1" for i in range(n_objectives, n_variables + 1)])

def dtlz5(n_variables: int, n_objectives: int) -> Problem:
    g_symbol = "g"
    g_expr = " + ".join([f"(x_{i} - 0.5)**2" for i in range(n_objectives, n_variables + 1)])
    g_expr = "1 + " + g_expr

    d_symbol = "d"
    # koska g_expr sisältää 1+g_expr, niin (1 + (2 * ({g_expr})) * x_{i}) lisättävä -1 ?
    d_expr = " * ".join([f"(1/{g_expr}) * (1 + (2 * ({g_expr})) * x_{i}) " for i in range(n_objectives, n_variables + 1)])
    d_expr = "1/2 * " + d_expr

    objectives = []
    for m in range(1, n_objectives + 1):
        # function f_m
        prod_expr = " * ".join([f"Cos(0.5 * {np.pi} * {d_expr})" for i in range(1, n_objectives - m + 1)])
        if m > 1:
            prod_expr += f"{' * ' if prod_expr != "" else ""}Sin(0.5 * {np.pi} * x_{d_expr})"
        if prod_expr == "":
            prod_expr = "1"  # When m == n_objectives, the product is empty, implying f_M = g.
        f_m_expr = f"({g_symbol}) * ({prod_expr})"

        objectives.append(
            Objective(
                name=f"f_{m}",
                symbol=f"f_{m}",
                func=f_m_expr,
                maximize=False,
                ideal=0,
                nadir=1, 
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=True,
            )
        )

    variables = [
        Variable(
            name=f"x_{i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.real,
            lowerbound=0,
            upperbound=1,
            initial_value=1.0,
        )
        for i in range(1, n_variables + 1)
    ]

    extras = [
        ExtraFunction(
            name="g", symbol=g_symbol, func=g_expr, is_convex=False, is_linear=False, is_twice_differentiable=True,
        ),
        ExtraFunction(
            name="d", symbol=d_symbol, func=d_expr, is_convex=False, is_linear=False, is_twice_differentiable=True
        ),
    ]

    return Problem(
        name="dtlz5",
        description="The DTLZ5 test problem.",
        variables=variables,
        objectives=objectives,
        extra_funcs=extras,
    )


def dtlz4(n_variables: int, n_objectives: int) -> Problem:
    alpha = 100 # this parameter differents this from dtlz2: does not work very well for some reason.. only finding extreme solutions.
    # function g
    g_symbol = "g"
    g_expr = " + ".join([f"(x_{i} - 0.5)**2" for i in range(n_objectives, n_variables + 1)])
    g_expr = "1 + " + g_expr

    objectives = []
    for m in range(1, n_objectives + 1):
        # function f_m
        prod_expr = " * ".join([f"Cos(0.5 * {np.pi} * (x_{i}**{alpha}))" for i in range(1, n_objectives - m + 1)])
        if m > 1:
            prod_expr += f"{' * ' if prod_expr != "" else ""}Sin(0.5 * {np.pi} * (x_{n_objectives - m + 1}**{alpha}))" # needed here or not?
        if prod_expr == "":
            prod_expr = "1"  # When m == n_objectives, the product is empty, implying f_M = g.
        f_m_expr = f"({g_symbol}) * ({prod_expr})"

        objectives.append(
            Objective(
                name=f"f_{m}",
                symbol=f"f_{m}",
                func=f_m_expr,
                maximize=False,
                ideal=0,
                nadir=1, 
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=True,
            )
        )

    variables = [
        Variable(
            name=f"x_{i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.real,
            lowerbound=0,
            upperbound=1,
            initial_value=1.0,
        )
        for i in range(1, n_variables + 1)
    ]

    extras = [
        ExtraFunction(
            name="g", symbol=g_symbol, func=g_expr, is_convex=False, is_linear=False, is_twice_differentiable=True
        ),
    ]

    return Problem(
        name="dtlz4",
        description="The DTLZ4 test problem.",
        variables=variables,
        objectives=objectives,
        extra_funcs=extras,
    )

# TODO: seems to work? but not good problem as thne have to use nevergrad
def kurosawe():
    objectives = []

    f_1_sum_expr = " + ".join([f" -10 * Exp ( -0.2 * Sqrt(x_{i}**2 + x_{i+1}**2) ) " for i in range(1,2)])
    #f_1 = f"{-10*np.exp(-0.2*np.sqrt(x[0]**2 + x[1]**2))+(-10*(np.exp(-0.2*np.sqrt(x[1]**2 + x[2]**2))))}"
    # i cant get this syntax right
    #f_2 += "".join([f"{np.abs(x[i])}**0.8 + 5 * {np.sin(x[i])}**3}" for i in range(0, 2)])
    #f_2 = np.sum(np.abs(x[i])**0.8 + 5 * np.sin(x[i]**3) for i in range(0,2))

    f_2 = " + ".join([f"Abs ( x_{i})**0.8 + 5 * Sin (x_{i}**3)" for i in range(1,3)])

    objectives.append(
                    Objective(
                name=f"f_{1}",
                symbol=f"f_{1}",
                func=f_1_sum_expr,
                maximize=False,
                ideal=-20,
                nadir=-14,  
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=False,
            )
    )
    objectives.append(
                Objective(
                name=f"f_{2}",
                symbol=f"f_{2}",
                func=f_2,
                maximize=False,
                ideal=-12,
                nadir=0.5,  
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=False,
            )
    )
    variables = [
        Variable(
            name=f"x_{i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.real,
            lowerbound=-5,
            upperbound=5,
            initial_value=1.0,
        )
        for i in range(1, 3)
    ]

    return Problem(
        name="kurosawe",
        description="The Kurosawe test problem.",
        variables=variables,
        objectives=objectives,
        #extra_funcs=extras,
    )


# again I dont know ideal and nadir and I don't know if differentiable etc.
def vehicle_crash():
    """The crash safety design problem with 3 objectives.

    Liao, X., Li, Q., Yang, X., Zhang, W. & Li, W. (2007).
    Multiobjective optimization for crash safety design of vehicles
    using stepwise regression model. Structural and multidisciplinary
    optimization, 35(6), 561-569. https://doi.org/10.1007/s00158-007-0163-x

    Arguments:
        var_iv (np.array): Optional, initial variable values. Must be between
            1 and 3. Defaults are [2, 2, 2, 2, 2].

    Returns:
        MOProblem: a problem object.
    """
    objectives = []
    # mass
    f_1 = f"1640.2823 + 2.3573285 * x_{1} + 2.3220035 * x_{2} + 4.5688768 * x_{3}+ 7.7213633 * x_{4}+ 4.4559504 * x_{5}"
    #ain
    f_2 = f"6.5856 + 1.15 * x_{1} - 1.0427 * x_{2}+ 0.9738 * x_{3}+ 0.8364 * x_{4}- 0.3695 * x_{1} * x_{4}+ 0.0861 * x_{1} * x_{5}+ 0.3628 * x_{2} * x_{4}- 0.1106 * x_{1} ** 2- 0.3437 * x_{3} ** 2 + 0.1764 * x_{4} ** 2"

    # Intrusion
    f_3 = f"-0.0551+ 0.0181 * x_{1}+ 0.1024 * x_{2}+ 0.0421 * x_{3}- 0.0073 * x_{1} * x_{2}+ 0.024 * x_{2} * x_{3}- 0.0118 * x_{2} * x_{4}- 0.0204 * x_{3} * x_{4}- 0.008 * x_{2} *x_{5}- 0.0241 * x_{2} ** 2+ 0.0109 * x_{4} ** 2"
    
    # do not know ideal & nadir
    objectives.append(
                    Objective(
                name=f"f_{1}",
                symbol=f"f_{1}",
                func=f_1,
                maximize=False,
                ideal=1600,
                nadir=1700,  
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=False,
            )
    )
    objectives.append(
                    Objective(
                name=f"f_{2}",
                symbol=f"f_{2}",
                func=f_2,
                maximize=False,
                ideal=4,
                nadir=8,  
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=False,
            )
    )
    objectives.append(
                    Objective(
                name=f"f_{3}",
                symbol=f"f_{3}",
                func=f_3,
                maximize=False,
                ideal=0,
                nadir=1,  
                is_convex=False,
                is_linear=False,
                is_twice_differentiable=False,
            )
    )

    objectives = [objective_1, objective_2, objective_3]

    x_1 = Variable("x_1", var_iv[0], 1.0, 3.0)
    x_2 = Variable("x_2", var_iv[1], 1.0, 3.0)
    x_3 = Variable("x_3", var_iv[2], 1.0, 3.0)
    x_4 = Variable("x_4", var_iv[3], 1.0, 3.0)
    x_5 = Variable("x_5", var_iv[4], 1.0, 3.0)

    variables = [x_1, x_2, x_3, x_4, x_5]

    problem = MOProblem(variables=variables, objectives=objectives)
