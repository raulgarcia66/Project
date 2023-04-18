using LinearAlgebra
using JuMP
using Gurobi
import Random
using Pipe
include("./generate_random_parameters.jl")
include("./solve.jl")
include("./plotting.jl")

num_subs = 2   # number of classes of subprocesses
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
Γ = 1 .- rand(num_sub)*0.05

###############################################################################
##### Deterministic
T = 3    # time horizon ≈ 6 weeks of treatment

B_union_vec_subs = Vector{Vector{Vector{Pair{Int, Float64}}}}[]
v_union_vec_subs = Vector{Vector{Vector{Pair{Int, Float64}}}}[]
BB_vec_subs = Vector{Vector{Vector{Float64}}}[]
vv_vec_subs = Vector{Vector{Vector{Float64}}}[]

for sub = 1:num_subs
    B_union_vec, v_union_vec, BB_vec, vv_vec = compute_deterministic_data(states, actions, B_max, P[sub], C[sub], R[sub], Γ[sub], T);
    push!(B_union_vec_subs, copy(B_union_vec))
    push!(v_union_vec_subs, copy(v_union_vec))
    push!(BB_vec_subs, copy(BB_vec))
    push!(vv_vec_subs, copy(vv_vec))
end

init_state_subs = [1,2]   # arbitrary

x_vals, obj_val = solve_UBAP(num_subs, init_state_subs, global_budget, BB_vec_subs, vv_vec_subs)
indices = extract_budget_index(x_vals, num_subs)
# @show x_vals[2,:]