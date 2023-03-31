using LinearAlgebra
using JuMP
using Gurobi
import Random
using Pipe
include("./generate_data.jl")
include("./solve.jl")

num_sub = 1   # number of classes of subprocesses
global_B = 100   # global budget
T = 5    # time horizon ≈ 6 weeks of treatment

num_states = 5   # will assume all subMDPs have same number of states
states = [i for i = 1:num_states]

num_actions = 4   # will assume all subMDPs have same number of actions
actions = [i for i = 1:num_actions]

# Variables to hold state and action 
# s = zeros(Int, M)   # hold state of system
# a = zeros(Int, M)   # hold action taken by system

# Initial state distribution
# Random.seed!(1)
# α = rand(num_states)
# α = α ./ sum(α)
# # sum(α)

# Create P, utilities, costs, rewards
P, C, U, R = generate_data_rand(num_sub);
# U_terminal = create_u(num_states, num_actions, seed = seed)
U_terminal = zeros(num_states)  # reward when no action is taken (e.g., at end of planning horizon); might not be used

# Discount factor
Random.seed!(1)
Γ = 1 .- rand(num_sub)*0.05

###############################################################################
# Run tests
include("./solve.jl")
sub = 1
B_union_vec, B_tilde_union_vec, v_union_vec, v_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec = compute_useful_budgets(states, actions, global_B, P[sub], C[sub], R[sub], Γ[sub], T);

for t in 1:T, i in states, a in actions
    plot_V(i, t, B_union_vec[t][i], v_union_vec[t][i])
end

t = 3
state = 4
action = 1

B_union_vec[t][state]
B_vec[t][state, action]

B_tilde_union_vec[t][state]

v_union_vec[t][state]
v_vec[t][state, action]
v_tilde_union_vec[t][state]

plot_V(state, t, B_union_vec[t][state], v_union_vec[t][state])