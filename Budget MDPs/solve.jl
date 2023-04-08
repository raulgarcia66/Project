using Pipe

"""
For a single budget MDP.

states: vector of states
actions: vector of actions
B_max: maximum budget at any time for the subMDP
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
    B_tilde_vec = Array{Vector{Float64}, 2}[]   # store potentially useful budgets B~ = [B~]_[i,a]; T elements
    B_vec = Array{Vector{Float64}, 2}[]   # store useful budgets B = [B]_[i,a] by pruning B~; T elements
    B_tilde_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store potentially useful budgets B~u = [B~u]_[i] as (action => budget) pairs by unioning over actions, ⋃_{a ∈ A} [B]_[i,a]; T elements
    B_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store useful budgets Bu = [Bu]_[i] as (action => budget) pairs by pruning B~u; T elements
    BB_vec = Vector{Vector{Float64}}[]   # store useful budgets BB = [BB]_[i] by extracting the budgets from the pairs in Bu; these are the final useful budgets for each state i; T+1 elements

    v_tilde_vec = Array{Vector{Float64},2}[]  # store corresponding values v~ = [v~]_[i,a] (= Vectors) for potentially useful budgets B~; T elements
    v_vec = Array{Vector{Float64},2}[]   # store corresponding values v = [v]_[i,a] (= Vectors) (= Vectors) for useful budgets B; T elements
    v_tilde_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store potentially useful values v~u = [v~u]_[i] as (action => value) pairs by unioning over actions, ⋃_{a ∈ A} [v]_[i,a]; T elements
    v_union_vec = Vector{Vector{Pair{Int, Float64}}}[]   # store useful values vu = [vu]_[i] as (action => value) pairs by pruning v~u; T elements
    vv_vec = Vector{Vector{Float64}}[]   # store useful values vv = [vv]_[i] by extracting the budgets from the pairs in vu; these are the final useful values for each state i; T+1 elements
    
    # σ_tilde_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ~ = [σ~]_[i,a] (= Vectors) corresponding to potentially useful budget assignments; T elements
    # σ_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ = [σ]_[i,a] (= Vectors) corresponding to useful budget assignments; T elements

    ## Initialize
    B_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(B_tilde_vec, B_tilde)
    B = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(B_vec, B)
    BB = [[0.0] for _ = 1:num_states]   # each state at "t = 0 stages to go" has the sole useful budget level of 0
    push!(BB_vec, BB)

    v_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(v_tilde_vec, v_tilde)
    v = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(v_vec, v)
    vv = [[0.0] for _ = 1:num_states]   # assume no value gained at "t = 0 stages to go"
    push!(vv_vec, vv)

    σ_tilde = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    σ = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    
    for t = 1:T
        println("t = $t \n") #BB = $BB \nBB length: $(length(BB)) \n vv = $vv \nvv length: $(length(vv))\n")
        println("About to compute B_tilde[i,a].\n")
        for i in states, a in actions
            S = compute_reachable_states(P, i, a)   # reachable states from i after taking action a
            B_tilde[i,a], σ_tilde[i,a] = compute_potentially_useful_budgets_state_action(i, a, BB, S, B_max, P, c, Γ)
            v_tilde[i,a] = compute_potentially_useful_values_state_action(i, a, vv, σ_tilde[i,a], S, P, r[i,a], Γ)
        end
        push!(B_tilde_vec, B_tilde)
        push!(v_tilde_vec, v_tilde)
        # push!(σ_tilde_vec, σ_tilde)

        println("About to compute B[i,a].\n")
        for i in states, a in actions
            B[i,a], v[i,a], σ[i,a] = compute_useful_budgets_state_action(B_tilde[i,a], v_tilde[i,a], σ_tilde[i,a])
        end
        push!(B_vec, B)
        push!(v_vec, v)
        # push!(σ_vec, σ)

        ## Take union of B over actions a
        # B_tilde_union[i] is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs for state i
        # v_tilde_union[i] is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs for state i
        println("About to compute B_tilde_union.\n")
        B_tilde_union = map(i -> 
                            begin
                                @pipe map(a -> map(b -> a => b, B[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        v_tilde_union = map(i ->
                            begin
                                @pipe map(a -> map(b -> a => b, v[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        push!(B_tilde_union_vec, B_tilde_union)
        push!(v_tilde_union_vec, v_tilde_union)

        # println("B_tilde_union: $(B_tilde_union)")
        # println("v_tilde_union: $(v_tilde_union)\n")

        ## Prune B_tilde_union to get B_union
        println("About to compute B_union.\n")
        B_union, v_union = copy(B_tilde_union), copy(v_tilde_union)
        for i in states
            B_union[i], v_union[i] = computed_useful_budgets_state(B_tilde_union[i], v_tilde_union[i])
            # B_union[i], v_union[i], σ_union[i] = computed_useful_budgets_state(B_tilde_union[i], v_tilde_union[i], σ_tilde_union[i])
        end
        push!(B_union_vec, B_union)
        push!(v_union_vec, v_union)
        # push!(σ_union_vec, σ_union)

        # println("B_union: $(B_union)")
        # println("v_union: $(v_union)\n")

        BB = map(B_union_state -> map(bevo -> bevo.second, B_union_state) , B_union)
        push!(BB_vec, BB)
        vv = map(v_union_state -> map(bevo -> bevo.second, v_union_state) , v_union)
        push!(vv_vec, vv)
    end

    # Order chronologically
    reverse!(B_union_vec)
    reverse!(BB_vec)
    reverse!(B_tilde_union_vec)
    reverse!(v_union_vec)
    reverse!(vv_vec)
    reverse!(v_tilde_union_vec)
    reverse!(B_vec)
    reverse!(B_tilde_vec)
    reverse!(v_vec)
    reverse!(v_tilde_vec)

    return B_union_vec, BB_vec, B_tilde_union_vec, v_union_vec, vv_vec, v_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec   # , σ_vec, σ_tilde_vec
end


function compute_reachable_states(P_sub, i, a)
    ϵ = 1E-5
    return filter(j -> P_sub[i,j,a] > ϵ, 1:size(P_sub,2))
end


"""
For a single budget MDP.

i: state
a: action
b: useful budgets for state-action (i,a)
S: set of reachable states
B_max: maximum budget at any time for the subMDP
P: P(i,j,a)
c: c(i,a)
Γ: discount factor
"""
function compute_potentially_useful_budgets_state_action(i, a, b, S, B_max, P, c, Γ)
    # b is vector of useful budgets for all states and action a, at time t
    # b[j] is a vector of useful budgets for future state j 

    assignments = compute_assignments(b[S])
    # For σ ∈ assignments, σ is a vector of assignments of the indices of elements of S
    # E.g., Let states = [1,2,3,4] and S = [2,3]. Suppose b[2] has 2 elements and b[3] 3 elements.
    # ⟹ assignments  = [[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]]

    p_u_b = map(σ -> c[i,a] + Γ * sum([ P[i, S[j], a] * b[S[j]][σ[j]] for j in eachindex(S) ]), assignments)
    indices = filter(ind -> p_u_b[ind] <= B_max, eachindex(p_u_b))
    p_u_b = p_u_b[indices]
    assignments = assignments[indices]

    return p_u_b, assignments
end


function compute_assignments(M)
    # M is a vector of useful budgets available for each reachable state
    # M[i] = [b_1, b_2,...,b_{n_i}] for state i, which has n_i useful budgets
    # We are taking combinations of their indices
    # E.g., Let states = [1,2,3,4] and S = [2,3]. Suppose b[2] has 2 elements and b[3] 3 elements.
    # ⟹ assignments  = [[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]]

    return @pipe vec(collect(Iterators.product([1:length(M[i]) for i in eachindex(M)]...))) |> map(a -> collect(a), _)
end


function compute_potentially_useful_values_state_action(i, a, v, assignments, S, P, r, Γ)
    # v[i] = vector of values (for each budget) for state i    
    return map(σ -> r + Γ * sum([ P[i, S[j], a] * v[S[j]][σ[j]] for j in eachindex(S) ]), assignments)
end


function compute_useful_budgets_state_action(b, v, σ)
    # b: set of potentially useful budgets, for some state and action
    # v: set of potentially useful values, for some state and action
    # σ: set of permutations for potentially useful budgets, for some state and action

    # println("b = $b \nv = $v \n σ = $σ \n")

    if !isempty(b)   # all should be nonempty if b is nonempty
        b = sort(b)
        sorted_ind = sortperm(b)
        v = v[sorted_ind]
        σ = σ[sorted_ind]

        # Prune budgets whose value is beat by the value of any previous smaller budget
        eps = 1E-5
        indices = filter(k ->
                begin
                    all( map(ℓ -> v[ℓ] < v[k] - eps, 1:(k-1)) )
                end, eachindex(b))

        return b[indices], v[indices], σ[indices]
    else
        return b, v, σ
    end
end

function computed_useful_budgets_state(B, v)
    # B is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs, for some state
    # v is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs, for some state

    if !isempty(B)
        sorted_ind = sortperm(B, by = a_b_pairs -> a_b_pairs.second)
        B = B[sorted_ind]
        v = v[sorted_ind]
        # σ = σ[sorted_ind]

        # Prune budgets whose value is beat by the value of any previous smaller budget
        eps = 1E-5
        indices = filter(k ->
                begin
                    all( map(ℓ -> v[ℓ].second < v[k].second - eps, 1:(k-1)) )
                end, eachindex(B))

        return B[indices], v[indices]#, σ[indices]
    else
        return B, v
    end
end


# struct Policy
#     B_union_vec
#     v_union_vec
#     T
#     action
#     # functions
# end


function get_action_budget_value(b, B_union_vec, v_union_vec, t, i)
    # return action for when budget is b
    # t is period
    # i is state

    ϵ = 1E-6
    index = findfirst(bud -> bud.second > b + ϵ, B_union_vec[t][i])   # find first budget that exceeds b
    if index === nothing
        index = length(B_union_vec[t][i])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previousb
    end
    return B_union_vec[t][i][index].first, B_union_vec[t][i][index].second, v_union_vec[t][i][index].second
end


## Functions for stochastic policies

# TODO: Swap B_union with BB if pairs not needed
function compute_Q_functions(B_union, v_union, states, actions, P)
    # B_union[i] = vector of budgets for state i
    # v_union[i] = vector of values for state i

    # B_ND_union, v_ND_union = compute_nondominated_budgets_values(B_union, v_union)
    B_union, v_union = compute_nondominated_budgets_values(B_union, v_union)

    BpB = Vector{Vector{Float64}}(undef, length(states))
    for i in states
        BpB[i] = compute_bang_per_buck(B_union[i], v_union[i]) # this is defined for budgets 2:end since the first budget is 0
    end

    # B_S_vec = Vector{Vector{Pair{Int,Float64}}}[]
    # B_S_vec = Vector{Pair{Int,Float64}}[]
    B_S = Array{Vector{Pair{Int, Tuple{Float64,Float64}}},2}(undef, length(states), length(actions))
    v_S = Array{Vector{Pair{Int, Tuple{Float64,Float64}}},2}(undef, length(states), length(actions))

    for i in states, a in actions

        S = compute_reachable_states(P, i, a)

        # TODO: Remove duplicates, since we're merging over states
        B_S[i,a] = @pipe map(j -> 
                    begin 
                        map(k -> j => (B_union[j][k], BpB[j][k-1]), 2:length(B_union[j]))   # budget of 0 excluded
                    end, S) |> vcat(_...)
        v_S[i,a] = @pipe map(j -> 
                    begin 
                        map(k -> j => (v_union[j][k], BpB[j][k-1]), 2:length(v_union[j]))   # budget of 0 excluded
                    end, S) |> vcat(_...)
        # println("B_S = $B_S")

        sorted_ind = sortperm(B_S[i,a], by = pairs -> pairs.second[2])
        B_S[i,a] = B_S[i,a][sorted_ind]
        v_S[i,a] = v_S[i,a][sorted_ind]
        # sort!(B_S[i,a], by = pairs -> pairs.second[2])
        # sort!(v_S[i,a], by = pairs -> pairs.second[2])
    end
    return B_S, v_S
end


# TODO: Figure this function out
function compute_nondominated_budgets_values(B_union, v_union)
    # B_union[i] = vector of budgets for state i
    # v_union[i] = vector of values for state i
    
    # Given useful budgets for state i, B[i], B[i][k] is said to dominated if ∃ two bracketing levels B[i][k-], B[i][k+] (k- < k < k+) such that
    # some convex combination of their values dominates that of B[i][k], i.e. (1-p)*v[i][k-] + p*v[i][k+] > v[i][k]
    return B_union, v_union
end


function compute_bang_per_buck(B, v)
    return map(k -> (v[k] - v[k-1]) / (B[k] - B[k-1]), 2:length(B))
end