# DOCUMENTATION FOR GNIMBUS v1


## Setting up the problem

Set up the problem normally and use nimbus generate_starting_point.


```
n_variables = 8
n_objectives = 3
problem = dtlz2(n_variables, n_objectives)
solver_options = IpoptOptions()
# get some initial solution
initial_rp = {
    "f_1": 0.4, "f_2": 0.5, "f_3": 0.8
}
initial_result = generate_starting_point(problem, initial_rp)
initial_fs = initial_result.optimal_objectives
initial_fs
```

## Learning phase
### The solution process starts with the learning phase. 

We iterate until the DMs wish to go to the decision phase or based on a "stopping" criterion. For example, let us run 3 iterations.
```

# for first iteration
next_current_solution = initial_result.optimal_objectives
print(f"initial solution: {next_current_solution}")

```

## ITERATION STARTS HERE

```
dms_rps = {
    "DM1": {"f_1": 0.0, "f_2": next_current_solution["f_2"], "f_3": 1},  # improve f_1, keep f_2 same, impair f_3
    "DM2": {"f_1": 0.3, "f_2": 1, "f_3": 0.5},  # improve f_1 to 0.3, impair f_2, improve f_3 to 0.5
    "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.0},  # impair f_1 to 0.5, impair f_2 to 0.6, improve f_3
}
```
### Run iteration

GNIMBUS solve_sub_problems solves the MOP by using different scalarization functions. Currently, the list involves group_nimbus, group_stom, group_asf and group_guess. Adding more scalarization functions to this can be done easily.

```
num_desired = 4
solutions = solve_sub_problems(
    problem, next_current_solution, dms_rps, num_desired, decision_phase=False, create_solver=PyomoIpoptSolver, solver_options=solver_options
)
for s in solutions:
    print(f"Solution: {s.optimal_objectives}")

gnimbus = solutions[0].optimal_objectives
gstom = solutions[1].optimal_objectives
gasf = solutions[2].optimal_objectives
gguess = solutions[3].optimal_objectives


```

### Select the next current iteration point by voting procedure
Voting procedure combines existing voting rules tried in the following order:
- Majority rule
- plurality rule
- if two solutions with most votes: we find intermediate solution
- if none above applies, we select the group_nimbus solution.


```
votes_idxs = {
        "DM1": 1,
        "DM2": 2,
        "DM3": 2,
}
voting_res = voting_procedure(problem, solutions, votes_idxs)
next_current_solution = voting_res.optimal_objectives
print("next current solution:", next_current_solution)

```
### Go to next iteration or move to decision phase
# "Decision phase"

``
dms_rps = {
    "DM1": {"f_1": 0.3, "f_2": 0.6, "f_3": 0.6},
    "DM2": {"f_1": 0.3, "f_2": 0.2, "f_3": 0.5},
    "DM3": {"f_1": 0.4, "f_2": 0.1, "f_3": 0.4},
}
```
### Let us only propose one solution found by group nimbus which respects each DMs bounds.

We set decision phase = True to only use group_nimbus. Currently the num desired does not matter, but later may be useful.

```
num_desired = 1 
solutions = solve_sub_problems(
    problem, next_current_solution, dms_rps, num_desired, decision_phase=True, create_solver=PyomoIpoptSolver, solver_options=solver_options
)
gnimbus_solution = solutions[0].optimal_objectives
print("Final solution candidate:", gnimbus_solution)

```
# Ask if the DMs agree upon the final solution. Otherwise go to group discussion.

# Group discussion:
Goal is to help the DMs understand what is going on and what can they do about it, for example, why the found solution does not differ too much from the current one and how they should adjust their preferences if they are not willing to stop with this solution but find something different.

- give DMs information about the state; what is achievable and what is not, unless preferences are changed. explain why not moving etc.
- Nudge or ask DMs to adjust their preferences to find better suiting solutions for the group.
- Finally, set next_current_solution to gnimbus_solution and go back to DMs giving preferences for the next iteration


```
next_current_solution = gnimbus_solution
```
