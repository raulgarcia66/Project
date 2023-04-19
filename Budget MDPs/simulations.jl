using LinearAlgebra
using JuMP
using Gurobi
import Random
using Pipe
include("./generate_random_parameters.jl")
include("./BMDP functions.jl")
include("./budget allocation.jl")
include("./simulation functions.jl")
include("./plotting.jl")

num_subs = 3   # number of classes of subprocesses
B_max = 10   # max budget at any states
global_budget = 10   # budget to be distributed among subclasses

num_states = 3   # will assume all subMDPs have same number of states
states = [i for i = 1:num_states]

num_actions = 2   # will assume all subMDPs have same number of actions
actions = [i for i = 1:num_actions]

# Initial state distribution
# Random.seed!(1)
# α = rand(num_states)
# α = α ./ sum(α)
# # sum(α)

# Create P, utilities, costs, rewards
P, C, U, R = generate_data_rand(num_subs, num_states, num_actions);
# U_terminal = create_u(num_states, num_actions, seed = seed)
U_terminal = zeros(num_states)  # reward when no action is taken (e.g., at end of planning horizon); might not be used

# Discount factor
Random.seed!(1)
Γ = 1 .- rand(num_subs)*0.05

###############################################################################
##### Deterministic
T = 3    # time horizon ≈ 6 weeks of treatment

B_union_vec_subs, v_union_vec_subs, σ_union_vec_subs, BB_vec_subs, vv_vec_subs, σ_vec_subs = compute_deterministic_data_multiple(num_subs, states, actions, B_max, P, C, R, Γ, T)

init_state_subs = rand(1:3,3)   # arbitrary

x_vals, obj_val = solve_UBAP(num_subs, init_state_subs, global_budget, BB_vec_subs, vv_vec_subs)
index_subs = extract_budget_index(x_vals, num_subs)
init_budget_subs = map(sub -> BB_vec_subs[sub][1][init_state_subs[sub]][index_subs[sub]], 1:num_subs)

budgets, actions_taken, states_visited, values_gained, costs_incurred = simulate_det(init_budget_subs, init_state_subs, 
                                                                            B_union_vec_subs, v_union_vec_subs, σ_union_vec_subs, BB_vec_subs, P, C, R, Γ, T);

actions_taken
budgets
states_visited

costs_incurred
@pipe sum(costs_incurred) |> sum(_)
values_gained
@pipe sum(values_gained) |> sum(_)
obj_val


budgets_vec = []
actions_taken_vec = []
states_visited_vec =[]
values_gained_vec = []
costs_incurred_vec = []
runs = 1000
for _ = 1:runs
    budgets, actions_taken, states_visited, values_gained, costs_incurred = simulate_det(init_budget_subs, init_state_subs, 
                                                                            B_union_vec_subs, v_union_vec_subs, σ_union_vec_subs, BB_vec_subs, P, C, R, Γ, T)
    push!(budgets_vec, budgets)
    push!(actions_taken_vec, actions_taken)
    push!(states_visited_vec, states_visited)
    push!(values_gained_vec, values_gained)
    push!(costs_incurred_vec, costs_incurred)
end

costs_vec = map(c_i -> sum(sum(c_i)), costs_incurred_vec)
values_vec = map(v_g -> sum(sum(v_g)), values_gained_vec)
using Statistics
mean(costs_vec)   # 16.893
sum(init_budget_subs)   # 9.996
mean(values_vec)   # 30.383
obj_val   # 30.494