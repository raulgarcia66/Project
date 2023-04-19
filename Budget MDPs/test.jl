using LinearAlgebra
using JuMP
using Gurobi
import Random
using Pipe
include("./generate_random_parameters.jl")
include("./BMDP functions.jl")
include("./plotting.jl")

num_subs = 1   # number of classes of subprocesses
B_max = 10   # max budget at any state; TODO: should not be synonymous with global budget
global_budget = 10   # budget to be distributed among subclasses

num_states = 3   # will assume all subMDPs have same number of states
states = [i for i = 1:num_states]

num_actions = 2   # will assume all subMDPs have same number of actions
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
P, C, U, R = generate_data_rand(num_subs, num_states, num_actions);
# U_terminal = create_u(num_states, num_actions, seed = seed)
U_terminal = zeros(num_states)  # reward when no action is taken (e.g., at end of planning horizon); might not be used

# Discount factor
Random.seed!(1)
Γ = 1 .- rand(num_subs)*0.05

###############################################################################
sub = 1
T = 3   # time horizon ≈ 6 weeks of treatment

B_union_vec, v_union_vec, σ_union_vec, BB_vec, vv_vec, σ_vec = compute_deterministic_data(states, actions, B_max, P[sub], C[sub], R[sub], Γ[sub], T);

state = 1
a = 1
b = 4.5
t = 1
act, bud, val = get_action_budget_value_det(b, state, B_union_vec, v_union_vec, t)
val = V_function_det(b, state, B_union_vec, v_union_vec, t)
val = V_function_det(b, state, BB_vec, vv_vec, t)

plot_V_det(state, t, BB_vec[t][state], vv_vec[t][state])
plot_V_det(state, t, B_union_vec[t][state], v_union_vec[t][state])

for t in 1:T+1, i in 1:num_states
    plot_V_det(i, t, BB_vec[t][i], vv_vec[t][i], save_plot=false)
end

for t = 1:T
    println("B_union (t=$t) = \n$(B_union_vec[t][state])\n")
    println("v_union (t=$t) = \n$(v_union_vec[t][state])\n")
    # println("B (t=$t) = \n$(B_vec[t][state])\n")
    # println("v (t=$t) = \n$(v_vec[t][state])\n")
    # if t == T
    #     println("B (t=$(t+1)) = \n$(B_vec[t+1][state])\n")
    #     println("v (t=$(t+1)) = \n$(v_vec[t+1][state])\n")
    # end
end

###############################################################################
sub = 1
T = 7
Q_star_vec, sto_B_V_vec, sto_v_V_vec, sto_B_vec, sto_v_vec = compute_stochastic_data(states, actions, P[sub], C[sub], R[sub], Γ[sub], T);
Q_star_vec[2][2]
sto_B_V_vec[2][2]
sto_v_V_vec[2][2]
sto_B_vec[2][2]
sto_v_vec[2][2]

state = 3
a = 1
b = 4.5
t = 1
Q_i_a_b_t = Q_function_stochastic(b, state, a, sto_B_vec[t], sto_v_vec[t])
Q_i_a_b_t = Q_function_stochastic(b, state, a, sto_B_vec, sto_v_vec, t)
V_i_b_t = V_function_stochastic(b, state, sto_B_V_vec[t], sto_v_V_vec[t])
V_i_b_t = V_function_stochastic(b, state, sto_B_V_vec, sto_v_V_vec, t)

plot_V_stochastic(state, t, sto_B_V_vec[t][state], sto_v_V_vec[t][state], save_plot=false)
plot_V_stochastic(state, sto_B_V_vec[t][state], sto_v_V_vec[t][state], save_plot=false)

plot_Q_stochastic(state, a, t, sto_B_vec[t][state,a], sto_v_vec[t][state,a], save_plot=false)
plot_Q_stochastic(state, a, sto_B_vec[t][state,a], sto_v_vec[t][state,a], save_plot=false)

p, exp_value, action_tup, budget_tup, value_tup = get_action_budget_value_stochastic(b, state, Q_star_vec, t)
action_lower, action_upper = action_tup
budget_lower, budget_upper = budget_tup
value_lower, value_upper = value_tup

for t in 1:T+1, i in 1:num_states
    plot_V_stochastic(i, t, sto_B_V_vec[t][i], sto_v_V_vec[t][i], save_plot=false)
end

for t = 1:T
    println("sto_B_V (s=$state, t=$t) = \n$(sto_B_V_vec[t][state])")
    println("sto_v_V (s=$state, t=$t) = \n$(sto_v_V_vec[t][state])\n")
    # println("sto_B (s=$state, t=$t) = \n$(sto_B_vec[t][state,:])")
    # println("sto_v (s=$state, t=$t) = \n$(sto_v_vec[t][state,:])\n")
    if t == T
        println("sto_B_V (s=$state, t=$(t+1)) = \n$(sto_B_V_vec[t+1][state])")
        println("sto_v_V (s=$state, t=$(t+1)) = \n$(sto_v_V_vec[t+1][state])\n")
    end
end

###############################################################################
t = 3
state = 1
action = 2
for t = 1:T
    println("B_union (t=$t) = \n$(B_union_vec[t][state])\n")
    println("v_union (t=$t) = \n$(v_union_vec[t][state])\n")
    # println("B (t=$t) = \n$(B_vec[t][state])\n")
    # println("v (t=$t) = \n$(v_vec[t][state])\n")
    # if t == T
    #     println("B (t=$(t+1)) = \n$(B_vec[t+1][state])\n")
    #     println("v (t=$(t+1)) = \n$(v_vec[t+1][state])\n")
    # end
end

for t in 1:T+1, i in 1:num_states
    plot_V_det(i, t, BB_vec[t][i], vv_vec[t][i], save_plot=true)
end

length(B_union_vec)
length(BB_vec)
length(B_tilde_union_vec)

length(B_tilde_vec)
length(B_vec)

length(v_union_vec)
length(vv_vec)
length(v_tilde_union_vec)

length(v_tilde_vec)
length(v_vec)

t = 3
state = 1
action = 2

B_union_vec[end][state]
B_vec[end][state, action]
B_tilde_union_vec[1][state]

v_union_vec[end][state]
v_vec[end][state, action]
v_tilde_union_vec[end][state]

###############################################################################
state = 3
a = 1
b = 1.2
t = 3
B_v_S, sto_B, sto_v = compute_Q_function_stochastic_data(BB_vec[t], vv_vec[t], states, actions, P[1], C[1], R[1], Γ[1]);
B_v_S[state,a]
sto_B[state,a]
sto_v[state,a]
Q_i_a_b = Q_function_stochastic(b, state, a, sto_B, sto_v)

Q_star, sto_B_V, sto_v_V = compute_V_function_stochastic_data_state(sto_B[state,:], sto_v[state,:])
Q_star
sto_B_V
sto_v_V

Q_star = Vector{Vector{Tuple{Int64, Float64, Float64}}}(undef, length(states))
sto_B_V = Vector{Vector{Float64}}(undef, length(states))
sto_v_V = Vector{Vector{Float64}}(undef, length(states))
for i in states
    Q_star[i], sto_B_V[i], sto_v_V[i] = compute_V_function_stochastic_data_state(sto_B[i,:], sto_v[i,:])
end
Q_star[1]
sto_B_V[1]
sto_v_V[1]