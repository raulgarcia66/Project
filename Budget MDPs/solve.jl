using Pipe

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
function compute_useful_budgets(states, actions, B_max, P, c, r, Γ, T)
    num_states = length(states)
    num_actions = length(actions)

    ## Store values for each iteration
    B_tilde_vec = Array{Vector{Float64}, 2}[]   # store potentially useful B = [B]_[i,a] for each time t; should have T+1 elements
    B_vec = Array{Vector{Float64}, 2}[]   # store useful B = [B]_[i,a] for each time t; should have T+1 elements
    B_tilde_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store potentially useful budgets B[i] by unioning over actions, ⋃_{a ∈ A} [B]_[i,a]
    B_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store useful budgets B[i] by unioning over actions, ⋃_{a ∈ A} [B]_[i,a]
    
    # v_tilde_vec = Array{Vector{Vector{Float64}},2}[]   # store corresponding values v(i,a) (= Vectors) for potentially useful budgets; should have T+1 elements
    # v_vec = Array{Vector{Vector{Float64}},2}[]   # store corresponding values v(i,a) (= Vectors) for useful budgets; should have T+1 elements
    v_tilde_vec = Array{Vector{Float64},2}[]
    v_vec = Array{Vector{Float64},2}[]
    v_tilde_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # TODO: describe these two
    v_union_vec = Vector{Vector{Pair{Int, Float64}}}[]
    
    σ_tilde_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ(i,a) (= Vectors) corresponding to potentially budget assignments; should have T elements
    σ_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ(i,a) (= Vectors) corresponding to budget assignments; should have T elements
    # σ_tilde_union_vec = Vector{Vector{Pair{Int, Vector{Int}}}}[]   # TODO: describe these two
    # σ_union_vec = Vector{Vector{Pair{Int, Vector{Int}}}}[]

    ## Initialize
    B_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # each state at "t = 0 stages to go" has the sole useful budget level of 0
    push!(B_tilde_vec, B_tilde)
    B = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    push!(B_vec, B)

    v_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # assume no value gained when finished
    push!(v_tilde_vec, v_tilde)
    v = [[0.0] for _ = 1:num_states, _ = 1:num_actions]   # no processing on v_tilde to get v at "t = 0 stages to go"
    push!(v_vec, v)

    σ_tilde = [[[1]] for _ = 1:num_states, _ = 1:num_actions]   # First mapping is computed in loop so nothing to push!() here
    σ = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    
    for _ = 1:T
        B_tilde_prev = copy(B_tilde)
        v_tilde_prev = copy(v_tilde)
        for i in states, a in actions
            B_tilde[i,a], σ_tilde[i,a] = compute_potentially_useful_budgets_state_action(i, a, B_tilde_prev[i,a], B_max, P, c, Γ)
            v_tilde[i,a] = compute_potentially_useful_values_state_action(i, a, v_tilde_prev, σ_tilde[i,a], P, r[i,a], Γ)
        end
        # v_tilde_prev = copy(v_tilde)
        push!(B_tilde_vec, B_tilde)
        push!(σ_tilde_vec, σ_tilde)
        push!(v_tilde_vec, v_tilde)

        for i in states, a in actions
            B[i,a], v[i,a], σ[i,a] = compute_useful_budgets_state_action(B_tilde[i,a], v_tilde[i,a], σ_tilde[i,a])
        end
        push!(B_vec, B)
        push!(σ_vec, σ)
        push!(v_vec, v)

        ## Take union of B over actions a
        # We do the following for keeping track because certain actions will get pruned
        # B[i] is a vector of (action => budget levels) pairs; B[i][j] is budget levels at state i when taking j-th action
        # B[i][j].first is the actual action, B[i][j].second is the budget levels
        # v[i] is a vector of (action => values) pairs; v[i][j] is value at state i when taking j-th action
        # v[i][j].first is the actual action, σ[i][j].second is the value
        # σ[i] is a vector of (action => assignments) pairs; σ[i][j] is assignments at state i when taking j-th action
        # σ[i][j].first is the actual action, σ[i][j].second is the assignment
        # Note that if states = [1,2,3,4] and S = [2,4], the assignment σ[i][j].second is a two-element vector where the first entry is the budget for state 2 and the second is for state 4
        # B_tilde_union[i] is the set of potentially useful budget levels stored as (action => budget) pairs for state i
        B_tilde_union = map(i -> 
                            begin
                                @pipe map(a -> map(b -> a => b, B[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        v_tilde_union = map(i ->
                            begin
                                @pipe map(a -> map(b -> a => b, v[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        # σ_tilde_union = map(i -> [a => σ[i,a] for a in actions], states)   # previous
        push!(B_tilde_union_vec, B_tilde_union)
        push!(v_tilde_union_vec, v_tilde_union)
        # push!(σ_tilde_union_vec, σ_tilde_union)

        ## Prune B_tilde_union to get B_union
        # B_union[i] is the set of useful budget levels stored as (action => budget) pairs for state i
        # B_union, v_union, σ_union = copy(B_tilde_union), copy(v_tilde_union), copy(σ_tilde_union)
        B_union, v_union = copy(B_tilde_union), copy(v_tilde_union)
        for i in states
            # B_union[i], v_union[i], σ_union[i] = computed_useful_budgets_state(B_tilde_union[i], v_tilde_union[i], σ_tilde_union[i])
            B_union[i], v_union[i] = computed_useful_budgets_state(B_tilde_union[i], v_tilde_union[i])
        end
        push!(B_union_vec, B_union)
        push!(v_union_vec, v_union)
        # push!(σ_union_vec, σ_union)
    end

    # reverse(B_tilde_vec)   # first element is now useful budgets for first chronological time period
    return B_union_vec, B_tilde_union_vec, v_union_vec, v_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec, σ_vec, σ_tilde_vec
    # return B_union_vec, B_tilde_union_vec, v_union_vec, v_tilde_union_vec, σ_union_vec, σ_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec, σ_vec, σ_tilde_vec
end


"""
For a single budget MDP.

i: state
a: action
b: useful budgets for state-action (i,a)
P: P(i,j,a)
c: c(i,a)
Γ: discount factor
"""
function compute_potentially_useful_budgets_state_action(i, a, b, B_max, P, c, Γ)
    # b is useful budgets for state i and action a at time t

    S = compute_S(P, i, a)   # reachable states from i after taking action a
    assignments = compute_assignments(length(b), S)

    # println("$(typeof(assignments))")
    p_u_b = map(σ -> c[i,a] + Γ * sum([ P[i,j,a] * b[σ[j]] for j in eachindex(S) ]), assignments)
    indices = filter(ind -> p_u_b[ind] <= B_max, eachindex(p_u_b))
    assignments = assignments[indices]

    return p_u_b, assignments
end


function compute_S(P_sub, i, a)
    ϵ = 1E-5
    return filter(j -> P_sub[i,j,a] > ϵ, 1:size(P_sub,2))
end


function compute_assignments(M, reachable_states)
    # M is number of useful budgets available

    # lists = [[i for i = 1:M] for _ = 1:num_states]
    # combs = @pipe collect(Iterators.product(lists...)) |> vec(_)
    return @pipe vec(collect(Iterators.product([[i for i = 1:M] for _ = 1:length(reachable_states)]...))) |> map(a -> collect(a), _)
end


function compute_potentially_useful_values_state_action(i, a, v, assignments, P, r, Γ)
    return map(σ -> r + Γ * sum([ P[i,j,a] * v[i,a][σ[j]] for j in eachindex(σ) ]), assignments)
end


function compute_useful_budgets_state_action(b, v, σ)
    b = sort(b)
    sorted_ind = sortperm(b)
    v = v[sorted_ind]
    σ = σ[sorted_ind]

    eps = 1E-5
    indices = filter(k ->
            begin
                all( map(ℓ -> v[ℓ] < v[k] - eps, 1:(k-1)) )
            end, eachindex(b))

    return b[indices], v[indices], σ[indices]
end

function computed_useful_budgets_state(B, v) #, σ)
    # B is a vector of (action => budget levels) pairs; B[j] is budget levels when taking j-th action
    # B[j].first is the actual action, B[j].second is the budget levels
    # v is a vector of (action => values) pairs; v[j] is value when taking j-th action
    # v[j].first is the actual action, σ[j].second is the value
    # σ is a vector of (action => assignments) pairs; σ[j] is assignments when taking j-th action
    # σ[j].first is the actual action, σ[j].second is the assignment
    # Note that if states = [1,2,3,4] and S = [2,4], the assignment σ[j].second is a two-element vector where the first entry is the budget for state 2 and the second is for state 4


    # For testing
    # B, v, σ = B_union[1], v_union[1], σ_union[1]
    # B, v = B_union[1], v_union[1]

    sorted_ind = sortperm(B, by = a_b_pairs -> a_b_pairs.second)
    B = B[sorted_ind]
    v = v[sorted_ind]
    # σ = σ[sorted_ind]

    eps = 1E-5
    indices = filter(k ->
            begin
                all( map(ℓ -> v[ℓ].second < v[k].second - eps, 1:(k-1)) )
            end, eachindex(B))

    return B[indices], v[indices]#, σ[indices]
end
