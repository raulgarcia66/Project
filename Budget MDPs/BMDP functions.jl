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
function compute_deterministic_data(states, actions, B_max, P, c, r, Γ, T)
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
    
    σ_tilde_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ~ = [σ~]_[i,a] (= Vectors) corresponding to potentially useful budget assignments; T elements
    σ_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ = [σ]_[i,a] (= Vectors) corresponding to useful budget assignments; T elements
    σ_tilde_union_vec = Vector{Vector{Pair{Int, Vector{Int}}}}[]   # store mappings σ~u = [σ~u]_[i,a] (= Vectors) corresponding to potentially useful budget assignments; T elements
    σ_union_vec = Vector{Vector{Pair{Int, Vector{Int}}}}[]   # store mappings σu = [σu]_[i,a] (= Vectors) corresponding to useful budget assignments; T elements
    sigma_vec = Vector{Vector{Vector{Int}}}[]   # store useful mappings sigma = [sigma]_[i] by extract the mappings from σu; these are the final useful; T elements
    # Each σ is a vector of assignments of the indices of elements of reachable next states S
    # E.g., Let states = [1,2,3,4] and S = [2,3]. Suppose b[2] has 2 elements and b[3] 3 elements.
    # ⟹ assignments = [[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]]

    ## Initialize
    B_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(B_tilde_vec, B_tilde)
    B = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(B_vec, B)
    BB = [[0.0] for _ = 1:num_states]   # each state at "t = 0 stages to go" has the sole useful budget level of 0
    push!(BB_vec, copy(BB))

    v_tilde = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(v_tilde_vec, v_tilde)
    v = [[0.0] for _ = 1:num_states, _ = 1:num_actions]
    # push!(v_vec, v)
    vv = [[0.0] for _ = 1:num_states]   # assume no value gained at "t = 0 stages to go"
    push!(vv_vec, vv)

    σ_tilde = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    # push!(σ_tilde_vec, σ_tilde)
    σ = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    # push!(σ_vec, σ)
    # sigma = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    # # push!(sigma_vec, sigma)
    
    for t = 1:T
        println("Iteration = $t \n")
        println("About to compute B_tilde[i,a].\n")
        for i in states, a in actions
            S = compute_reachable_states(P, i, a)   # reachable states from i after taking action a
            B_tilde[i,a], σ_tilde[i,a] = compute_potentially_useful_budgets_state_action(i, a, BB, S, B_max, P, c, Γ)
            v_tilde[i,a] = compute_potentially_useful_values_state_action(i, a, vv, σ_tilde[i,a], S, P, r[i,a], Γ)
        end
        push!(B_tilde_vec, copy(B_tilde))
        push!(v_tilde_vec, copy(v_tilde))
        push!(σ_tilde_vec, copy(σ_tilde))

        println("About to compute B[i,a].\n")
        for i in states, a in actions
            B[i,a], v[i,a], σ[i,a] = compute_useful_budgets_values_state_action(B_tilde[i,a], v_tilde[i,a], σ_tilde[i,a])
        end
        push!(B_vec, copy(B))
        push!(v_vec, copy(v))
        push!(σ_vec, copy(σ))

        ## Take union of B over actions a
        # B_tilde_union[i] is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs for state i
        # v_tilde_union[i] is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs for state i
        # σ_tilde_union[i] is the set of potentially useful budget mappings after unioning over actions, stored as (action => mapping) pairs for state i
        println("About to compute B_tilde_union.\n")
        # TODO: Remove duplicates?
        B_tilde_union = map(i -> 
                            begin
                                @pipe map(a -> map(b -> a => b, B[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        v_tilde_union = map(i ->
                            begin
                                @pipe map(a -> map(v -> a => v, v[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        σ_tilde_union = map(i ->
                            begin
                                @pipe map(a -> map(s -> a => s, σ[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        push!(B_tilde_union_vec, copy(B_tilde_union))
        push!(v_tilde_union_vec, copy(v_tilde_union))
        push!(σ_tilde_union_vec, copy(σ_tilde_union))

        # println("B_tilde_union: $(B_tilde_union)")
        # println("v_tilde_union: $(v_tilde_union)")
        # println("σ_tilde_union: $(σ_tilde_union)\n")

        ## Prune B_tilde_union to get B_union
        println("About to compute B_union.\n")
        B_union, v_union, σ_union = copy(B_tilde_union), copy(v_tilde_union), copy(σ_tilde_union)
        for i in states
            # B_union[i], v_union[i] = compute_useful_budgets_values_state(B_tilde_union[i], v_tilde_union[i])
            B_union[i], v_union[i], σ_union[i] = compute_useful_budgets_values_state(B_tilde_union[i], v_tilde_union[i], σ_tilde_union[i])
        end
        push!(B_union_vec, copy(B_union))
        push!(v_union_vec, copy(v_union))
        push!(σ_union_vec, copy(σ_union))

        # println("B_union: $(B_union)")
        # println("v_union: $(v_union)\n")

        BB = map(B_union_state -> map(bevo -> bevo.second, B_union_state) , B_union)
        push!(BB_vec, copy(BB))
        vv = map(v_union_state -> map(bevo -> bevo.second, v_union_state) , v_union)
        push!(vv_vec, copy(vv))
        sigma = map(σ_union_state -> map(bevo -> bevo.second, σ_union_state) , σ_union)
        push!(sigma_vec, copy(sigma))
    end

    # Order chronologically
    reverse!(B_tilde_union_vec)
    reverse!(B_union_vec)
    reverse!(BB_vec)
    reverse!(v_tilde_union_vec)
    reverse!(v_union_vec)
    reverse!(vv_vec)
    reverse!(σ_tilde_union_vec)
    reverse!(σ_union_vec)
    reverse!(sigma_vec)

    reverse!(B_vec)
    reverse!(B_tilde_vec)
    reverse!(v_vec)
    reverse!(v_tilde_vec)
    reverse!(σ_vec)
    reverse!(σ_tilde_vec)

    # return B_union_vec, BB_vec, B_tilde_union_vec, v_union_vec, vv_vec, v_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec   # , σ_vec, σ_tilde_vec
    return B_union_vec, v_union_vec, σ_union_vec, BB_vec, vv_vec, sigma_vec
end


function compute_reachable_states(P, i, a)
    ϵ = 1E-5
    return filter(j -> P[i,j,a] > ϵ, 1:size(P,2))
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


function compute_useful_budgets_values_state_action(b, v, σ)
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


function compute_useful_budgets_values_state(B, v, σ)
    # B is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs, for some state
    # v is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs, for some state
    # σ is the set of potentially useful budget assignments after unioning over actions, stored as (action => budget) pairs, for some state

    # println("B = $B \nv = $v \n σ = $σ \n")

    if !isempty(B)   # all should be nonempty if b is nonempty
        sorted_ind = sortperm(B, by = a_b_pairs -> a_b_pairs.second)
        B = B[sorted_ind]
        v = v[sorted_ind]
        σ = σ[sorted_ind]

        # Prune budgets whose value is beat by the value of any previous smaller budget
        eps = 1E-5
        indices = filter(k ->
                begin
                    all( map(ℓ -> v[ℓ].second < v[k].second - eps, 1:(k-1)) )
                end, eachindex(B))

        return B[indices], v[indices], σ[indices]
    else
        return B, v, σ
    end
end

function compute_useful_budgets_values_state(B, v)
    # B is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs, for some state
    # v is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs, for some state

    if !isempty(B)
        sorted_ind = sortperm(B, by = a_b_pairs -> a_b_pairs.second)
        B = B[sorted_ind]
        v = v[sorted_ind]

        # Prune budgets whose value is beat by the value of any previous smaller budget
        eps = 1E-5
        indices = filter(k ->
                begin
                    all( map(ℓ -> v[ℓ].second < v[k].second - eps, 1:(k-1)) )
                end, eachindex(B))

        return B[indices], v[indices]
    else
        return B, v
    end
end


function Q_function_det(b, i, a, B_vec, v_vec, t)
    # Return value for when budget is b and action is A
    # t is period, i is state
    
    ϵ = 1E-8
    index = findfirst(bud -> bud > b + ϵ, B_vec[t][i,a])   # find first budget that exceeds b
    if index === nothing
        index = length(B_vec[t][i,a])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previous
    end
    return v_vec[t][i,a][index]
end


function V_function_det(b, i, B_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, v_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, t)
    # return action for when budget is b
    # t is period
    # i is state

    ϵ = 1E-6
    index = findfirst(bud -> bud.second > b + ϵ, B_union_vec[t][i])   # find first budget that exceeds b
    if index === nothing
        index = length(B_union_vec[t][i])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previous
    end
    return v_union_vec[t][i][index].second
end


function V_function_det(b, i, BB_vec::Vector{Vector{Vector{Float64}}}, vv_vec::Vector{Vector{Vector{Float64}}}, t)
    # return action for when budget is b
    # t is period
    # i is state

    ϵ = 1E-6
    index = findfirst(bud -> bud > b + ϵ, BB_vec[t][i])   # find first budget that exceeds b
    if index === nothing
        index = length(BB_vec[t][i])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previous
    end
    return vv_vec[t][i][index]
end


function get_action_budget_value_det(b, i, B_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, v_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, t)
    # return action for when budget is b
    # t is period
    # i is state

    ϵ = 1E-6
    index = findfirst(bud -> bud.second > b + ϵ, B_union_vec[t][i])   # find first budget that exceeds b
    if index === nothing
        index = length(B_union_vec[t][i])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previous
    end
    return B_union_vec[t][i][index].first, B_union_vec[t][i][index].second, v_union_vec[t][i][index].second
end


function get_action_budget_value_mapping_det(b, i, B_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, v_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, σ_union_vec::Vector{Vector{Vector{Pair{Int,Vector{Int}}}}}, t)
    # return action for when budget is b
    # t is period
    # i is state

    ϵ = 1E-6
    index = findfirst(bud -> bud.second > b + ϵ, B_union_vec[t][i])   # find first budget that exceeds b
    if index === nothing
        index = length(B_union_vec[t][i])   # budget is the last element
    elseif index != 1   # first budget is 0
        index -= 1   # our budget is the previous
    end
    return B_union_vec[t][i][index].first, B_union_vec[t][i][index].second, v_union_vec[t][i][index].second, σ_union_vec[t][i][index].second
end


function gen_next_state(P, current_state, action)
    U = rand()
    sum = 0
    for i in eachindex(P[current_state, :, action])
        sum += P[current_state, i, action]
        if sum > U
            return i
        end
    end
end

#################################################################################
##### Functions for stochastic policies

function compute_stochastic_data(states, actions, P, c, r, Γ, T)
    # Store data
    B_v_S_vec = Matrix{Vector{Tuple{Int64, Int64, Float64, Float64, Float64, Float64, Float64}}}[]
    sto_B_vec = Matrix{Vector{Float64}}[]
    sto_v_vec =  Matrix{Vector{Float64}}[]
    
    Q_star_vec = Vector{Vector{Tuple{Int64, Float64, Float64}}}[]
    sto_B_V_vec = Vector{Vector{Float64}}[]
    sto_v_V_vec = Vector{Vector{Float64}}[]

    # Initiate data
    B_v_S = Array{Vector{Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64}},2}(undef, length(states), length(actions))
    sto_B = Array{Vector{Float64},2}(undef, length(states), length(actions))
    sto_v = Array{Vector{Float64},2}(undef, length(states), length(actions))
    Q_star = Vector{Vector{Tuple{Int64, Float64, Float64}}}(undef, length(states))
    # If we copy from BB/vv_vec, would need remove nondominated points as well
    # sto_B_V = copy(BB_vec[end])
    # sto_v_V = copy(vv_vec[end])
    sto_B_V = [[0.0] for _ = 1:length(states)]
    push!(sto_B_V_vec, copy(sto_B_V))
    sto_v_V = [[0.0] for _ = 1:length(states)]
    push!(sto_v_V_vec, copy(sto_v_V))

    for t = 1:T
        println("Iteration = $t \n")
        println("About to compute sto_B and sto_v.\n")
        # println("sto_B_V (t = $t) = before\n$sto_B_V")
        # println("sto_v_V (t = $t) = before\n$sto_v_V\n")
        B_v_S, sto_B, sto_v = compute_Q_function_stochastic_data(sto_B_V, sto_v_V, states, actions, P, c, r, Γ)
        push!(B_v_S_vec, copy(B_v_S))
        push!(sto_B_vec, copy(sto_B))
        push!(sto_v_vec, copy(sto_v))
        # println("sto_B (t = $t) = \n$sto_B")
        # println("sto_v (t = $t) = \n$sto_v\n")

        for i in states
            Q_star[i], sto_B_V[i], sto_v_V[i] = compute_V_function_stochastic_data_state(sto_B[i,:], sto_v[i,:])
        end
        push!(Q_star_vec, copy(Q_star))
        push!(sto_B_V_vec, copy(sto_B_V))
        push!(sto_v_V_vec, copy(sto_v_V))
        # println("sto_B_V (t = $t) = after\n$sto_B_V")
        # println("sto_v_V (t = $t) = after\n$sto_v_V\n\n\n")
    end

    reverse!(B_v_S_vec)
    reverse!(sto_B_vec)
    reverse!(sto_v_vec)
    reverse!(Q_star_vec)
    reverse!(sto_B_V_vec)
    reverse!(sto_v_V_vec)

    return Q_star_vec, sto_B_V_vec, sto_v_V_vec, sto_B_vec, sto_v_vec # B_v_S_vec
end


function compute_Q_function_stochastic_data(B::Vector{Vector{Float64}}, v::Vector{Vector{Float64}}, states, actions, P, c, r, Γ)
    # B[i] = vector of budgets for state i
    # v[i] = vector of values for state i
    # Must be ordered by increasing budget
    B, v = compute_nondominated_budgets_values_Q(B, v)
    # for i in states
    #     plot_V_stochastic(i, B[i], v[i])
    # end

    BpB = Vector{Vector{Float64}}(undef, length(states))
    ΔB = Vector{Vector{Float64}}(undef, length(states))
    Δv = Vector{Vector{Float64}}(undef, length(states))
    for i in states
        BpB[i], ΔB[i], Δv[i] = compute_bang_per_buck(B[i], v[i]) # this is defined for budgets [2:end] since the first budget is 0
    end

    B_v_S = Array{Vector{Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64}},2}(undef, length(states), length(actions))
    sto_B = Array{Vector{Float64},2}(undef, length(states), length(actions))
    sto_v = Array{Vector{Float64},2}(undef, length(states), length(actions))

    for i in states, a in actions

        S = compute_reachable_states(P, i, a)

        # TODO: Remove duplicates, since we're merging over states
        B_v_S[i,a] = @pipe map(j -> 
                    begin 
                        map(k -> (j, k-1, B[j][k], v[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(B[j]))   # budget of 0 excluded
                        # Fix sort!() and sto_B/sto_v if I use bottom line with action inserted
                        # map(k -> (j, k-1, B_union[j][k].first, B[j][k], v[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(BB[j]))   # budget of 0 excluded
                    end, S) |> vcat(_...)
        sort!(B_v_S[i,a], by = t -> t[5], rev=true)

        # For the m-th element of sorted B_v_S[i,a]
            # j(m): state j, i.e., state j in b^{j}_k   # B_v_S[i,a][m][1]
            # k(m): the useful budget index for j, i.e., the k of b^{j}_k   # B_v_S[i,a][m][2]. Prior to sorting, this is index for BpB, ΔB and Δv, while index+1 is for B_union and v_union
            # Δb(m): budget increment   # # B_v_S[i,a][m][6]
            # Δv(m): value increment   # B_v_S[i,a][m][7]
        sto_B_0 = c[i,a]
        sto_B[i,a] = map(m -> sto_B_0 + sum([ P[i, B_v_S[i,a][ℓ][1], a] * B_v_S[i,a][ℓ][6] for ℓ = 1:m ]), eachindex(B_v_S[i,a]))
        pushfirst!(sto_B[i,a], sto_B_0)

        # TODO: Make sure both of these are correct
        sto_v_0 = r[i,a] + Γ * sum([ P[i, j, a] * v[j][1] for j in S ])
        sto_v[i,a] = map(m -> r[i,a] + Γ * (sto_v_0 + sum([ P[i, B_v_S[i,a][ℓ][1], a] * B_v_S[i,a][ℓ][7] for ℓ = 1:m ])), eachindex(B_v_S[i,a]))
        pushfirst!(sto_v[i,a], sto_v_0)

        ## Previous
        # B_S[i,a] = @pipe map(j -> 
        #             begin 
        #                 map(k -> (j, k-1) => (B_union[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(B_union[j]))   # budget of 0 excluded
        #             end, S) |> vcat(_...)
        # v_S[i,a] = @pipe map(j -> 
        #             begin 
        #                 map(k -> (j, k-1) => (v_union[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(v_union[j]))   # budget of 0 excluded
        #             end, S) |> vcat(_...)

        # sorted_ind = sortperm(B_S[i,a], by = pairs -> pairs.second[2], rev=true)
        # B_S[i,a] = B_S[i,a][sorted_ind]
        # v_S[i,a] = v_S[i,a][sorted_ind]
        # # sort!(B_S[i,a], by = pairs -> pairs.second[2], rev=true)
        # # sort!(v_S[i,a], by = pairs -> pairs.second[2], rev=true)

        # For the m-th element of sorted B_S[i,a]
            # j(m): state j, i.e., state j in b^{j}_k   # B_S[i,a][m].first[1]
            # k(m): the useful budget index for j, i.e., the k of b^{j}_k   # B_S[i,a][m].first[2]. Prior to sorting, this is index for BpB, ΔB and Δv, while index+1 is for B_union and v_union
            # Δb(m): budget increment   # # B_S[i,a][m].second[3]
            # Δv(m): value increment   # B_S[i,a][m].second[4]
    end

    return B_v_S, sto_B, sto_v
end


function compute_V_function_stochastic_data_state(sto_B, sto_v)
    # sto_B[a] = vector of budgets for action a, for some state
    # sto_v[a] = vector of values for action a, for some state

    # Create Q_star to prune points
    Q_star = @pipe map(a -> 
                        begin
                            map(k -> (a, sto_B[a][k], sto_v[a][k]), eachindex(sto_B[a]))
                        end, eachindex(sto_B)) |> vcat(_...)
    sort!(Q_star, by = tup -> tup[2])   # order in increasing budgets
    Q_star_final, sto_B_V_i, sto_v_V_i = compute_nondominated_budgets_values_V(Q_star)

    return Q_star_final, sto_B_V_i, sto_v_V_i

    # sto_B_i = map(tup -> tup[2], Q_star)   # nondominated
    # sto_v_i = map(tup -> tup[3], Q_star)   # nondominated

    # # Graham scan
    # Q_star_final = [Q_star[1]]
    # sto_B_V_i = [sto_B_i[1]]
    # sto_v_V_i = [sto_v_i[1]]
    # for j in 2:length(Q_star)
    #     last_point_not_deleted = false
    #     while !(last_point_not_deleted)
    #         if length(sto_B_V_i) > 1
    #             if (sto_v_i[j] - sto_v_V_i[end]) / (sto_B_i[j] - sto_B_V_i[end]) >=
    #                     (sto_v_V_i[end] - sto_v_V_i[end-1]) / (sto_B_V_i[end] - sto_B_V_i[end-1])
    #                 pop!(Q_star_final)
    #                 pop!(sto_B_V_i)
    #                 pop!(sto_v_V_i)
    #                 last_point_not_deleted = false
    #                 println("Point removed in compute_V_function_stochastic_data_state")
    #             else
    #                 push!(Q_star_final, Q_star[j])
    #                 push!(sto_B_V_i, sto_B_i[j])
    #                 push!(sto_v_V_i, sto_v_i[j])
    #                 last_point_not_deleted = true
    #             end
    #         else
    #             push!(Q_star_final, Q_star[j])
    #             push!(sto_B_V_i, sto_B_i[j])
    #             push!(sto_v_V_i, sto_v_i[j])
    #             last_point_not_deleted = true
    #         end
    #     end
    # end

    # return Q_star_final, sto_B_V_i, sto_v_V_i
end


function compute_nondominated_budgets_values_Q(B::Vector{Vector{Float64}}, v::Vector{Vector{Float64}})
    # B[i] = vector of budgets for state i
    # v[i] = vector of values for state i
    
    # Given useful budgets for state i, B[i], B[i][k] is said to dominated if ∃ two bracketing levels B[i][k-], B[i][k+] (k- < k < k+) such that
    # some convex combination of their values dominates that of B[i][k], i.e. (1-p)*v[i][k-] + p*v[i][k+] > v[i][k]
    # return B, v
    B_nd = Vector{Vector{Float64}}(undef, length(B))
    v_nd = Vector{Vector{Float64}}(undef, length(v))

    for i in eachindex(B)
        B_nd[i] = [B[i][1]]
        v_nd[i] = [v[i][1]]
        # Graham scan
        for j in 2:length(B[i])
            last_point_not_deleted = false
            while !(last_point_not_deleted)
                if length(B_nd[i]) > 1
                    if (v[i][j] - v_nd[i][end]) / (B[i][j] - B_nd[i][end]) >=
                            (v_nd[i][end] - v_nd[i][end-1]) / (B_nd[i][end] - B_nd[i][end-1])
                        pop!(B_nd[i])
                        pop!(v_nd[i])
                        last_point_not_deleted = false
                        # println("Point removed in compute_nondominated_budgets_values_Q")
                    else
                        push!(B_nd[i], B[i][j])
                        push!(v_nd[i], v[i][j])
                        last_point_not_deleted = true
                    end
                else
                    push!(B_nd[i], B[i][j])
                    push!(v_nd[i], v[i][j])
                    last_point_not_deleted = true
                end
            end
        end
    end

    return B_nd, v_nd
end


function compute_bang_per_buck(B, v)
    Δv = map(k -> (v[k] - v[k-1]), 2:length(B))
    ΔB = map(k -> (B[k] - B[k-1]), 2:length(B))
    return Δv ./ ΔB, ΔB, Δv
    # return map(k -> (v[k] - v[k-1]) / (B[k] - B[k-1]), 2:length(B))
end


function compute_nondominated_budgets_values_V(Q_star::Vector{Tuple{Int,Float64,Float64}})
    # Q_star is vector of (action, budget, value) tuples for some state, sorted in increasing order of budget
    # This function won't be nondecreasing in budget (ignoring actions)
    
    # Useful budget Q_star[k][2] is said to dominated if ∃ two different budget points Q_star[k-][2] ≤  Q_star[k][2] ≤ Q_star[k+][2] (k- < k < k+) such that
    # (1-α)*Q_star[k-][3] + α*Q_star[k+][3] > Q_star[k][3], where α = b - b_{k-} / b_{k+} - b_{k-}
    # return Q_star

    sto_B_i = map(tup -> tup[2], Q_star)   # nondominated
    sto_v_i = map(tup -> tup[3], Q_star)   # nondominated

    # Graham scan
    Q_star_final = [Q_star[1]]
    sto_B_V_i = [sto_B_i[1]]
    sto_v_V_i = [sto_v_i[1]]
    for j in 2:length(Q_star)
        last_point_not_deleted = false
        while !(last_point_not_deleted)
            if length(sto_B_V_i) > 1
                if (sto_v_i[j] - sto_v_V_i[end]) / (sto_B_i[j] - sto_B_V_i[end]) >=
                        (sto_v_V_i[end] - sto_v_V_i[end-1]) / (sto_B_V_i[end] - sto_B_V_i[end-1])
                    pop!(Q_star_final)
                    pop!(sto_B_V_i)
                    pop!(sto_v_V_i)
                    last_point_not_deleted = false
                    # println("Point removed in compute_nondominated_budgets_values_V")
                else
                    push!(Q_star_final, Q_star[j])
                    push!(sto_B_V_i, sto_B_i[j])
                    push!(sto_v_V_i, sto_v_i[j])
                    last_point_not_deleted = true
                end
            else
                push!(Q_star_final, Q_star[j])
                push!(sto_B_V_i, sto_B_i[j])
                push!(sto_v_V_i, sto_v_i[j])
                last_point_not_deleted = true
            end
        end
    end

    return Q_star_final, sto_B_V_i, sto_v_V_i
end


function Q_function_stochastic(b, i, a, sto_B, sto_v)

    if b < sto_B[i,a][1]
        error("Budget $b is less than the smallest useful budget.")
    elseif b > sto_B[i,a][end]
        return sto_v[i,a][end]
    end

    ind_eq = findfirst(==(b), sto_B[i,a])
    if ind_eq !== nothing
        return sto_v[i,a][ind_eq]
    else
        ind_g = findfirst(>(b), sto_B[i,a]) - 1
        p = (b - sto_B[i,a][ind_g]) / (sto_B[i,a][ind_g+1] - sto_B[i,a][ind_g])
        return p * sto_v[i,a][ind_g+1] + (1-p) * sto_v[i,a][ind_g]
    end
end


function Q_function_stochastic(b, i, a, sto_B_vec, sto_v_vec, t)

    if b < sto_B_vec[t][i,a][1]
        error("Budget $b is less than the smallest useful budget.")
    elseif b > sto_B_vec[t][i,a][end]
        return sto_v_vec[t][i,a][end]
    end

    ind_eq = findfirst(==(b), sto_B_vec[t][i,a])
    if ind_eq !== nothing
        return sto_v_vec[t][i,a][ind_eq]
    else
        ind_g = findfirst(>(b), sto_B_vec[t][i,a]) - 1
        p = (b - sto_B_vec[t][i,a][ind_g]) / (sto_B_vec[t][i,a][ind_g+1] - sto_B_vec[t][i,a][ind_g])
        return p * sto_v_vec[t][i,a][ind_g+1] + (1-p) * sto_v_vec[t][i,a][ind_g]
    end
end


function V_function_stochastic(b, i, sto_B_V, sto_v_V)
    # Return expected value of V(i,b) for some period

    if b < sto_B_V[i][1]
        error("Budget $b is less than the smallest useful budget.")
    elseif b > sto_B_V[i][end]
        return sto_v_V[i][end]
    end

    ind_eq = findfirst(==(b), sto_B_V[i])
    if ind_eq !== nothing
        return sto_v_V[i][ind_eq]
    else
        ind_g = findfirst(>(b), sto_B_V[i]) - 1
        p = (b - sto_B_V[i][ind_g]) / (sto_B_V[i][ind_g+1] - sto_B_V[i][ind_g])
        return p * sto_v_V[i][ind_g+1] + (1-p) * sto_v_V[i][ind_g]
    end
end


function V_function_stochastic(b, i, sto_B_V_vec, sto_v_V_vec, t)
    # Return expected value of V(i,b) at time t

    if b < sto_B_V_vec[t][i][1]
        error("Budget $b is less than the smallest useful budget.")
    elseif b > sto_B_V_vec[t][i][end]
        return sto_v_V_vec[t][i][end]
    end

    ind_eq = findfirst(==(b), sto_B_V_vec[t][i])
    if ind_eq !== nothing
        return sto_v_V_vec[t][i][ind_eq]
    else
        ind_g = findfirst(>(b), sto_B_V_vec[t][i]) - 1
        p = (b - sto_B_V_vec[t][i][ind_g]) / (sto_B_V_vec[t][i][ind_g+1] - sto_B_V_vec[t][i][ind_g])
        return p * sto_v_V_vec[t][i][ind_g+1] + (1-p) * sto_v_V_vec[t][i][ind_g]
    end
end


# Just for V function? Q as well?
function get_action_budget_value_stochastic(b, i, Q_star_vec, t)
    # Return expected value of V(i,b) at time t
    # Q_star_vec[t][i][k] is tuple (action, budget, value) of value function V(i,b) at period t
    budgets = map(tup -> tup[2], Q_star_vec[t][i])

    if b < budgets[1]
        error("Budget $b is less than the smallest useful budget.")
    elseif b > budgets[end]
        p = 1
        exp_value = Q_star_vec[t][i][end][3]
        action_lower = Q_star_vec[t][i][end][1]
        action_upper = Q_star_vec[t][i][end][1]
        budget_lower = Q_star_vec[t][i][end][2]
        budget_upper = Q_star_vec[t][i][end][2]
        value_lower = Q_star_vec[t][i][end][3]
        value_upper = Q_star_vec[t][i][end][3]
        return p, exp_value, (action_lower, action_upper), (budget_lower, budget_upper), (value_lower, value_upper)
    end

    ind_eq = findfirst(==(b), budgets)
    if ind_eq !== nothing
        p = 1
        exp_value = Q_star_vec[t][i][ind_eq][3]
        action_lower = Q_star_vec[t][i][ind_eq][1]
        action_upper = Q_star_vec[t][i][ind_eq][1]
        budget_lower = Q_star_vec[t][i][ind_eq][2]
        budget_upper = Q_star_vec[t][i][ind_eq][2]
        value_lower = Q_star_vec[t][i][ind_eq][3]
        value_upper = Q_star_vec[t][i][ind_eq][3]
        return p, exp_value, (action_lower, action_upper), (budget_lower, budget_upper), (value_lower, value_upper)
    else
        ind_g = findfirst(>(b), budgets) - 1
        p = (b - budgets[ind_g]) / (budgets[ind_g+1] - budgets[ind_g])
        exp_value = p * Q_star_vec[t][i][ind_g+1][3] + (1-p) * Q_star_vec[t][i][ind_g][3]
        action_lower = Q_star_vec[t][i][ind_g][1]
        action_upper = Q_star_vec[t][i][ind_g+1][1]
        budget_lower = Q_star_vec[t][i][ind_g][2]
        budget_upper = Q_star_vec[t][i][ind_g+1][2]
        value_lower = Q_star_vec[t][i][ind_g][3]
        value_upper = Q_star_vec[t][i][ind_g+1][3]
        return p, exp_value, (action_lower, action_upper), (budget_lower, budget_upper), (value_lower, value_upper)
    end
end


function gen_action(p, action_lower, action_upper, budget_lower, budget_upper, value_lower, value_upper)
    U = rand()
    if p <= U
        return action_lower, budget_lower, value_lower
    else
        return action_upper, budget_upper, value_upper
    end
end


function gen_action(p, action_tup::NTuple{2,Int}, budget_tup::NTuple{2,Float64}, value_tup::NTuple{2,Float64})
    action_lower, action_upper = action_tup
    budget_lower, budget_upper = budget_tup
    value_lower, value_upper = value_tup
    return gen_action(p, action_lower, action_upper, budget_lower, budget_upper, value_lower, value_upper)
end

# # Julia note 
# x = [[1,2,3]]
# y = [[4,5]]
# push!(y, x[1])
# x[1][3] = 6
# x
# y   # y is also updated