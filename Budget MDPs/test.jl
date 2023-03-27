using LinearAlgebra
using JuMP
using Gurobi
import Random
using Pipe

M = 5   # number of classes of subprocesses
B = 100   # global budget
T = 30    # time horizon ≈ 6 weeks of treatment

num_states = 5   # will assume all subMDPs have same number of states
states = [i for i = 1:num_states]

num_actions = 4   # will assume all subMDPs have same number of actions
actions = [i for i = 1:num_actions]

# Variables to hold state and action 
s = zeros(Int, M)   # hold state of system
a = zeros(Int, M)   # hold action taken by system

Random.seed!(1)
α = rand(num_states)   # Initial state distribution
α = α ./ sum(α)
# sum(α)

# Create P, utilities, costs, rewards
P = Array{Float64,3}[]
C = Array{Float64,2}[]
U = Array{Float64,2}[]
R = Array{Float64,2}[]
# U_terminal = create_u(num_states, num_actions, seed = seed)
U_terminal = zeros(num_states)  # reward when no action is taken (e.g., at end of planning horizon); might not be used

for seed = 1:M
    push!(P, create_P_mat(num_states, num_actions, seed = seed))

    ## Create r independently
    # push!(r, create_r(num_states, num_actions, seed = seed))
    ## Create r from u and c
    local u, c
    u = create_u(num_states, num_actions, seed = seed)
    push!(U, u)
    c = create_c(num_states, num_actions, seed = seed)
    push!(C, c)
    push!(R, create_r(u, c))
end

# Discount factor
Random.seed!(1)
Γ = 1 .- rand(N)*0.05

###############################################################################
# Test functions
p = create_P_mat(num_states, num_actions, seed=1)
r = create_r(num_states, num_actions, seed=1)

u = create_u(num_states, num_actions, seed = 1)
c = create_c(num_states, num_actions, seed = 1)
r = create_r(u, c)

###############################################################################

function create_P_mat(num_states, num_actions; seed=-1)
    if seed != -1
        Random.seed!(seed)
    end

    P = zeros(num_states, num_states, num_actions)
    for i = 1:num_states
        for k = 1:num_actions
            row = rand(num_states)
            P[i,:,k] = row ./ sum(row)
        end
    end

    return P
end


function create_r(num_states, num_actions; seed=-1)
    if seed != -1
        Random.seed!(seed)
    end

    return rand(-1:10, num_states, num_actions)
end


function create_r(u, c)
    return u - c
end


function create_u(num_states, num_actions; seed=-1)
    if seed != -1
        Random.seed!(seed)
    end

    u_col = rand(0:5, num_states)
    u = copy(u_col)
    for _ = 1:(num_actions-1)
        u = hcat(u, u_col)
    end

    return u
end


function create_c(num_states, num_actions; seed=-1)
    if seed != -1
        Random.seed!(seed)
    end

    c = rand(1:5, num_states, num_actions)
    c[:, end] .= 0   # need one free action
    # if !(any(x -> x == 0, c))
    #     ind = rand(eachindex(c))
    #     c[ind] = 0
    # end

    # return rand(0:5, num_states, num_actions)
    return c
end
