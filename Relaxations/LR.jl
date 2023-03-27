using LinearAlgebra
using JuMP
using Gurobi
# import XLSX
import Random
using Pipe

M = 10   # number of subprocesses
N = 1   # number of linking constraints
num_states = 5   # will assume all subMDPs have same number of states
num_actions = 4   # will assume all subMDPs have same number of actions

s = zeros(Int, M)   # hold state of system
a = zeros(Int, M)   # hold action taken by system

Random.seed!(100)
α = rand(num_states)   # Initial state distribution
α = α ./ sum(α)
# sum(α)

# Create P and r
P = Matrix{Float64}[]
r = Matrix{Float64}[]

for seed = 1:M
    push!(P, create_P_mat(num_states, num_actions, seed = seed))
    push!(r, create_r(num_states, num_actions, seed = seed))
end

# Discount factor
Random.seed!(1)
β = 1 .- rand(N)*0.05

# Linking constraints. Each row corresponds to a resouce, each column to a subprocess
D_vec = []
for i = 1:M
    D_i = Array{Float64, 3}(undef, num_states, num_actions, N)
    for state in 1:num_states, action in 1:num_actions
        D_i[state, action, :] = ones(N)   # vector of dim N
    end
    push!(D_vec, D_i)
end

b = zeros(N)
# Process b

###############################################################################

# function create_P_mat(num_states; seed=-1)
#     if seed != -1
#         Random.seed!(seed)
#     end

#     P = zeros(num_states, num_states)
#     for i = 1:num_states
#         row = rand(num_states)
#         P[i,:] = row ./ sum(row)
#     end

#     return P
# end

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

    return rand(-1:5, num_states, num_actions)
end





# ############################## Import data ##################################
# data_xlsx = XLSX.readxlsx("C:/Users/rauli/OneDrive/Documents/Rice University/CAAM 654/Project/MDP transitions_updated.xlsx")
# data_sheet = data_xlsx["Sheet1"]

# # Probability transition matrix
# p = data_sheet["A2:J11"]
# for i = 1:size(p,1)
#     for j = 1:size(p,2)
#         if p[i,j] === missing
#             p[i,j] = 0
#         end
#     end
# end
# p = Matrix{Float64}(p)

# push!(P, p)

# # Rewards pre- and post- transplant (weeks)
# R_post = Matrix{Float64}(data_sheet["A14:J14"])
# R_pre = Matrix{Float64}(data_sheet["A19:J19"])

# # Discount factor
# λ = Float64(data_sheet["B22"])

# # Other data
# num_states = size(P,1)
# states = [i for i = 1:num_states]   # last state is death
# # actions = ["T"; "W"]

# # Three algorithms for solving infinite-horizon MDPs