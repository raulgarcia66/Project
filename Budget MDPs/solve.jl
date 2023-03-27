"""
For a single budget MDP.

states:
actions:
P: P(i,j,a)
c: c(i,a)
r: r(i,a)
Γ: discount factor
T: horizon
"""
function compute_useful_budgets(states, actions, P, c, r, Γ, T)
    num_states = length(states)
    num_actions = length(actions)

    ## Store values for each iteration
    # B_tilde = Array{Vector{Float64}, 2}(undef, num_states, num_actions)   # B[s,a] = Vector{Float64} of useful budgets
    B_tilde_vec = Array{Vector{Float64}, 2}[]   # store potentially useful B = [B]_[s,a] for each time t; should have T+1 elements
    B_vec = Array{Vector{Float64}, 2}[]   # store useful B = [B]_[s,a] for each time t; should have T+1 elements
    σ_vec = Array{Vector{Vector{Float64}}, 2}[]   # store mappings σ(i,a) (= Vectors) corresponding to budget assignments; should have T elements
    v_tilde_vec = Array{Vector{Vector{Float64}},2}[]   # store corresponding values v(i,a) (= Vectors) for potentially useful budgets; should have T+1 elements
    v_vec = Array{Vector{Vector{Float64}},2}[]   # store corresponding values v(i,a) (= Vectors) for useful budgets; should have T+1 elements

    ## Initialize
    B_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # each state at "t = 0 stages to go" has the sole useful budget level of 0
    push!(B_tilde_vec, B_tilde)
    B = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # each state at "t = 0 stages to go" has the sole useful budget level of 0
    push!(B_vec, B) ##### Push?
    v_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # assume no value gained when finished
    push!(v_tilde_vec, v)
    v = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # assume no value gained when finished
    push!(v_vec, v) ##### Push?
    
    for _ = 1:T+1
        for i in states, a in actions
            B_tilde[i,a], σ_tild = compute_potentially_useful_budgets_state_action(i, a, B_tilde[i,a], P, c, Γ)
            v[i,a] = compute_potentially_useful_values_state_action(i, a, v, σ, P, r[i,a], Γ)
        end
        push!(B_tilde_vec, B_tilde)
        push!(σ_vec, σ)
        push!(v_tilde_vec, v)

        for i in states, a in actions
            # TODO: Find permutation of σ as well. Might need σ_tilde
            B[i,a], v[i,a] = compute_useful_budgets_state_action(B_tilde[i,a], v[i,a])
        end
        push!(B_tilde_vec, B_tilde)
        push!(σ_vec, σ)
        push!(v_tilde_vec, v)
    end

    # reverse(B_tilde_vec)   # first element is now useful budgets for first chronological time period
end


function compute_potentially_useful_budgets_state_action(i, a, b, P, c, Γ)
    # b is useful budgets for state i and action a at time t

    S = compute_S(P, i, a)   # reachable states from i after taking action a
    perms = compute_permutations(length(b), length(S))

    p_u_b = map(σ -> c(i,a) + Γ * sum([ P[i,j,a] * b[σ[j]] for j in S ]), perms)
    indices = filter(ind -> p_u_b[ind] <= B_max, eachindex(p_u_b))
    perms = perms[indices]

    return p_u_b, perms
end


function compute_S(P_sub, i, a)
    ϵ = 1E-5
    return filter(j -> P_sub[i,j,a] > ϵ, 1:size(P_sub,2))
end


function compute_permutations(M, num_states)
    # TODO
end


function compute_potentially_useful_values_state_action(i, a, v, perms, P, r, Γ)
    return map(perm -> r + Γ * sum([ P[i,j,a] * v[i,j][perm[j]] for j = axes(P,2) ]), perms)
end


function compute_useful_budgets_state_action(b, v)
    b = sort(b)
    sorted_ind = sortperm(b)
    v = v[sorted_ind]

    eps = 1E-5
    indices = filter(k ->
            begin
                all( map(ℓ -> v[ℓ] < v[k] - eps, 1:(k-1)) )
            end, eachindex(b))

    return b[indices], v[indices]
end


# function potentially_useful_budgets_state_action(i, a, S, b, P, c, Γ; B_max=Inf)
#     # σ_vec = Int[]
#     perms = compute_permutations(length(b), length(S))

#     p_u_b = map(σ -> c(i,a) + Γ*sum([ P[i,j,a] * b[σ[j]] for j in S ]), perms)
#     indices = filter(ind -> p_u_b[ind] <= B_max, eachindex(p_u_b))
#     perms = perms[indices]

#     # p_u_b = map(index -> c(i,a) + sum([ P[i,j,a] * b[perm[index][j]] for j in S ]), eachindex(perms))   # potentially useful budgets for previous period
#     # indices = filter(index -> c(i,a) + sum([ P[i,j,a] * b[perm[index][j]] for j in S ]) <= B_max, eachindex(perms))
#     # perms = perms[indices]

#     return p_u_b, perms
# end


# """
# For a single budget MDP.

# i: state
# a: action
# t: t stages to go
# P: P(i,j,a)
# c: c(i,a)
# Γ: discount factor
# """
# function compute_potentially_useful_budgets_state_action(i, a, b, P, c, Γ)
#     # b is useful budgets for state i and action a at time t

#     S = compute_S(P, i, a)   # reachable states from i after taking action a
#     p_u_b, σ = potentially_useful_budgets_state_action(i, a, S, b, P, c, Γ)

# end