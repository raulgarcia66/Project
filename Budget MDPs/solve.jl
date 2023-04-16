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
    
    # σ_tilde_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ~ = [σ~]_[i,a] (= Vectors) corresponding to potentially useful budget assignments; T elements
    # σ_vec = Array{Vector{Vector{Int}}, 2}[]   # store mappings σ = [σ]_[i,a] (= Vectors) corresponding to useful budget assignments; T elements

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
    σ = [[[1]] for _ = 1:num_states, _ = 1:num_actions]
    
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
        # push!(σ_tilde_vec, σ_tilde)

        println("About to compute B[i,a].\n")
        for i in states, a in actions
            B[i,a], v[i,a], σ[i,a] = compute_useful_budgets_values_state_action(B_tilde[i,a], v_tilde[i,a], σ_tilde[i,a])
        end
        push!(B_vec, copy(B))
        push!(v_vec, copy(v))
        # push!(σ_vec, σ)

        ## Take union of B over actions a
        # B_tilde_union[i] is the set of potentially useful budgets after unioning over actions, stored as (action => budget) pairs for state i
        # v_tilde_union[i] is the set of potentially useful values after unioning over actions, stored as (action => budget) pairs for state i
        println("About to compute B_tilde_union.\n")
        # TODO: Remove duplicates?
        B_tilde_union = map(i -> 
                            begin
                                @pipe map(a -> map(b -> a => b, B[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        v_tilde_union = map(i ->
                            begin
                                @pipe map(a -> map(b -> a => b, v[i,a]), actions) |> collect(Iterators.flatten(_))
                            end, states)
        push!(B_tilde_union_vec, copy(B_tilde_union))
        push!(v_tilde_union_vec, copy(v_tilde_union))

        # println("B_tilde_union: $(B_tilde_union)")
        # println("v_tilde_union: $(v_tilde_union)\n")

        ## Prune B_tilde_union to get B_union
        println("About to compute B_union.\n")
        B_union, v_union = copy(B_tilde_union), copy(v_tilde_union)
        for i in states
            B_union[i], v_union[i] = computed_useful_budgets_values_state(B_tilde_union[i], v_tilde_union[i])
            # B_union[i], v_union[i], σ_union[i] = computed_useful_budgets_state(B_tilde_union[i], v_tilde_union[i], σ_tilde_union[i])
        end
        push!(B_union_vec, copy(B_union))
        push!(v_union_vec, copy(v_union))
        # push!(σ_union_vec, σ_union)

        # println("B_union: $(B_union)")
        # println("v_union: $(v_union)\n")

        BB = map(B_union_state -> map(bevo -> bevo.second, B_union_state) , B_union)
        push!(BB_vec, copy(BB))
        vv = map(v_union_state -> map(bevo -> bevo.second, v_union_state) , v_union)
        push!(vv_vec, copy(vv))
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

    # return B_union_vec, BB_vec, B_tilde_union_vec, v_union_vec, vv_vec, v_tilde_union_vec, B_vec, B_tilde_vec, v_vec, v_tilde_vec   # , σ_vec, σ_tilde_vec
    return B_union_vec, v_union_vec, BB_vec, vv_vec
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

function computed_useful_budgets_values_state(B, v)
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


# TODO:
# function Q_function_det(b, i, B_vec, v_vec, t)

# end


function V_function_det(b, i, B_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, v_union_vec::Vector{Vector{Vector{Pair{Int,Float64}}}}, t)
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
        index -= 1   # our budget is the previousb
    end
    return vv_vec[t][i][index]
end


function get_action_budget_value_det(b, i, B_union_vec, v_union_vec, t)
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


#################################################################################
## Functions for stochastic policies

# TODO: Add t as parameter? Lets see how the induction step is done
function compute_V_function_stochastic_data(BB_vec, vv_vec, states, actions, P, c, r, Γ, T)
    # num_states = length(states)
    # num_actions = length(actions)

    # Store data
    B_v_S_vec = Matrix{Vector{Tuple{Int64, Int64, Float64, Float64, Float64, Float64, Float64}}}[]
    sto_B_vec = Matrix{Vector{Float64}}[]
    sto_v_vec =  Matrix{Vector{Float64}}[]
    
    Q_star_vec = Vector{Vector{Tuple{Int64, Float64, Float64}}}[]
    sto_B_V_vec = Vector{Vector{Float64}}[]
    sto_v_V_vec = Vector{Vector{Float64}}[]

    # Initiate data
    Q_star = Vector{Vector{Tuple{Int64, Float64, Float64}}}(undef, length(states))
    # sto_B_V = Vector{Vector{Float64}}(undef, length(states))
    # sto_v_V = Vector{Vector{Float64}}(undef, length(states))
    B_v_S = Array{Vector{Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64}},2}(undef, length(states), length(actions))
    sto_B = Array{Vector{Float64},2}(undef, length(states), length(actions))
    sto_v = Array{Vector{Float64},2}(undef, length(states), length(actions))

    # TODO: Remove nondominated points here too
    sto_B_V = copy(BB_vec[end])
    push!(sto_B_V_vec, copy(sto_B_V))
    sto_v_V = copy(vv_vec[end])
    push!(sto_v_V_vec, copy(sto_v_V))

    # for t = T+1:-1:1
    for t = T:-1:1
        # TODO: Should be feeding in budgets/values of new value function with respect to the stochastic method?
        # B_v_S, sto_B, sto_v = compute_Q_function_stochastic_data(BB_vec[t], vv_vec[t], states, actions, P, c, r, Γ)
        println("sto_B_V (t = $t) = before\n$sto_B_V")
        println("sto_v_V (t = $t) = before\n$sto_v_V\n")
        B_v_S, sto_B, sto_v = compute_Q_function_stochastic_data(sto_B_V, sto_v_V, states, actions, P, c, r, Γ)
        push!(B_v_S_vec, copy(B_v_S))
        push!(sto_B_vec, copy(sto_B))
        push!(sto_v_vec, copy(sto_v))
        println("sto_B (t = $t) = \n$sto_B")
        println("sto_v (t = $t) = \n$sto_v\n")

        for i in states
            Q_star[i], sto_B_V[i], sto_v_V[i] = compute_V_function_stochastic_data_state(sto_B[i,:], sto_v[i,:])
        end
        push!(Q_star_vec, copy(Q_star))
        push!(sto_B_V_vec, copy(sto_B_V))
        push!(sto_v_V_vec, copy(sto_v_V))
        println("sto_B_V (t = $t) = after\n$sto_B_V")
        println("sto_v_V (t = $t) = after\n$sto_v_V\n\n\n")
    end

    reverse!(B_v_S_vec)
    reverse!(sto_B_vec)
    reverse!(sto_v_vec)
    reverse!(Q_star_vec)
    reverse!(sto_B_V_vec)
    reverse!(sto_v_V_vec)

    # TODO: reverse vector for chronology as appropriate
    return Q_star_vec, sto_B_V_vec, sto_v_V_vec, sto_B_vec, sto_v_vec
end


function compute_Q_function_stochastic_data(BB, vv, states, actions, P, c, r, Γ)
    # # B_union[i] = vector of (action => budgets) for state i
    # # v_union[i] = vector of (action => values) for state i

    # B_union, v_union = compute_nondominated_budgets_values_Q(B_union, v_union)
    # BB = map(B_union_state -> map(bevo -> bevo.second, B_union_state) , B_union)
    # vv = map(v_union_state -> map(bevo -> bevo.second, v_union_state) , v_union)


    # BB[i] = vector of budgets for state i
    # vv[i] = vector of values for state i
    BB, vv = compute_nondominated_budgets_values_Q(BB, vv)

    BpB = Vector{Vector{Float64}}(undef, length(states))
    ΔB = Vector{Vector{Float64}}(undef, length(states))
    Δv = Vector{Vector{Float64}}(undef, length(states))
    for i in states
        BpB[i], ΔB[i], Δv[i] = compute_bang_per_buck(BB[i], vv[i]) # this is defined for budgets [2:end] since the first budget is 0
    end

    B_v_S = Array{Vector{Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64}},2}(undef, length(states), length(actions))
    # B_S = Array{Vector{Pair{Tuple{Int,Int}, NTuple{4,Float64}}},2}(undef, length(states), length(actions))
    # v_S = Array{Vector{Pair{Tuple{Int,Int}, NTuple{4,Float64}}},2}(undef, length(states), length(actions))
    sto_B = Array{Vector{Float64},2}(undef, length(states), length(actions))
    sto_v = Array{Vector{Float64},2}(undef, length(states), length(actions))

    for i in states, a in actions

        S = compute_reachable_states(P, i, a)

        # TODO: Remove duplicates, since we're merging over states
        B_v_S[i,a] = @pipe map(j -> 
                    begin 
                        map(k -> (j, k-1, BB[j][k], vv[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(BB[j]))   # budget of 0 excluded
                        # Fix sort!() and sto_B/sto_v if use bottom line with action
                        # map(k -> (j, k-1, B_union[j][k].first, BB[j][k], vv[j][k], BpB[j][k-1], ΔB[j][k-1], Δv[j][k-1]), 2:length(BB[j]))   # budget of 0 excluded
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

        sto_v_0 = r[i,a] + Γ * sum([ P[i, j, a] * vv[j][1] for j in S ])
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

    return B_v_S, sto_B, sto_v   # B_S, v_S
end


# TODO: Add t as parameter? Lets see how the induction step is done
function compute_V_function_stochastic_data_state(sto_B, sto_v)
    # sto_B[a] = vector of budgets for action a, for some state
    # sto_v[a] = vector of values for action a, for some state

    # Create Q_star to prune points
    Q_star = @pipe map(a -> 
                        begin
                            map(k -> (a, sto_B[a][k], sto_v[a][k]), eachindex(sto_B[a]))
                        end, eachindex(sto_B)) |> vcat(_...)
    sort!(Q_star, by = tup -> tup[2])   # order in increasing budgets
    Q_star = compute_nondominated_budgets_values_V(Q_star)

    # TODO: Keep this vector?
    actions_Q_star = map(tup -> tup[1], Q_star)

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


function compute_nondominated_budgets_values_Q(B_union, v_union)
    # B_union[i] = vector of budgets for state i
    # v_union[i] = vector of values for state i
    
    # Given useful budgets for state i, B[i], B[i][k] is said to dominated if ∃ two bracketing levels B[i][k-], B[i][k+] (k- < k < k+) such that
    # some convex combination of their values dominates that of B[i][k], i.e. (1-p)*v[i][k-] + p*v[i][k+] > v[i][k]
    return B_union, v_union
end


function compute_bang_per_buck(B, v)
    Δv = map(k -> (v[k] - v[k-1]), 2:length(B))
    ΔB = map(k -> (B[k] - B[k-1]), 2:length(B))
    return Δv ./ ΔB, ΔB, Δv
    # return map(k -> (v[k] - v[k-1]) / (B[k] - B[k-1]), 2:length(B))
end


# TODO: Figure this function out. Reuse above function if possible
function compute_nondominated_budgets_values_V(Q_star)
    # Q_star is vector of (action, budget, value) tuples for some state sorted in increasind order of budget
    # This function won't be nondecreasing in budget (ignoring actions)
    
    # Useful budget Q_star[k][2] is said to dominated if ∃ two different budget points Q_star[k-][2] ≤  Q_star[k][2] ≤ Q_star[k+][2] (k- < k < k+) such that
    # (1-α)*Q_star[k-][3] + α*Q_star[k+][3] > Q_star[k][3], where α = b - b_{k-} / b_{k+} - b_{k-}
    return Q_star
end


# TODO: Add t as parameter? Lets see how the induction step is done
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


end

function get_action_budget_value_stochastic(b, i, B_union_vec, v_union_vec, t)

end