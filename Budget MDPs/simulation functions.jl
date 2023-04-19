function compute_deterministic_data_multiple(num_subs, states, actions, B_max, P, C, R, Γ, T)
    B_union_vec_subs = Vector{Vector{Vector{Pair{Int, Float64}}}}[]
    v_union_vec_subs = Vector{Vector{Vector{Pair{Int, Float64}}}}[]
    σ_union_vec_subs = Vector{Vector{Vector{Pair{Int, Vector{Int}}}}}[]
    BB_vec_subs = Vector{Vector{Vector{Float64}}}[]
    vv_vec_subs = Vector{Vector{Vector{Float64}}}[]
    σ_vec_subs = Vector{Vector{Vector{Vector{Int}}}}[]

    for sub = 1:num_subs
        # B_union_vec, v_union_vec, BB_vec, vv_vec = compute_deterministic_data(states, actions, B_max, P[sub], C[sub], R[sub], Γ[sub], T);
        B_union_vec, v_union_vec, σ_union_vec, BB_vec, vv_vec, σ_vec = compute_deterministic_data(states, actions, B_max, P[sub], C[sub], R[sub], Γ[sub], T);
        push!(B_union_vec_subs, copy(B_union_vec))
        push!(v_union_vec_subs, copy(v_union_vec))
        push!(σ_union_vec_subs, copy(σ_union_vec))
        push!(BB_vec_subs, copy(BB_vec))
        push!(vv_vec_subs, copy(vv_vec))
        push!(σ_vec_subs, copy(σ_vec))
    end
    return B_union_vec_subs, v_union_vec_subs, σ_union_vec_subs, BB_vec_subs, vv_vec_subs, σ_vec_subs
end


function simulate_det(init_budget_subs, init_state_subs, B_union_vec_subs, v_union_vec_subs, σ_union_vec_subs, BB_vec_subs, P, C, R, Γ, T)
    
    num_subs = length(P)
    
    # Variables for each iteration
    current_budget = copy(init_budget_subs)
    current_state = copy(init_state_subs)
    current_action = zeros(Int, num_subs)

    # Store info for each time period
    # actions_taken = Vector{Vector{Int}}(undef, T)
    # budgets = Vector{Vector{Float64}}(undef, T)
    actions_taken = [zeros(Int,num_subs) for _ in 1:T]
    budgets = [zeros(num_subs) for _ in 1:T]
    states_visited = [zeros(Int, num_subs) for _ in 1:T]
    values_gained = [zeros(num_subs) for _ in 1:T]
    costs_incurred = [zeros(num_subs) for _ in 1:T]

    for sub in 1:num_subs
        println("Assigned budget at t = 1 for sub $sub in state $(current_state[sub]):    \n$(current_budget[sub])\n")
    end

    for t = 1:T
        for sub in 1:num_subs
            local v, σ
            current_action[sub], current_budget[sub], v, σ = get_action_budget_value_mapping_det(current_budget[sub], current_state[sub], 
                                                                                                B_union_vec_subs[sub], v_union_vec_subs[sub], σ_union_vec_subs[sub], t)
            states_visited[t][sub] = copy(current_state[sub])
            actions_taken[t][sub] = copy(current_action[sub])
            budgets[t][sub] = copy(current_budget[sub])

            # TODO: This correct?
            costs_incurred[t][sub] += Γ[sub]^(t-1) * C[sub][current_state[sub], current_action[sub]]
            values_gained[t][sub] += Γ[sub]^(t-1) * R[sub][current_state[sub], current_action[sub]]

            # Update current state and budget with next period's
            next_state = gen_next_state(P[sub], current_state[sub], current_action[sub])
            S = compute_reachable_states(P[sub], current_state[sub], current_action[sub])
            next_state_ind = findfirst(==(next_state), S)
            current_budget[sub] = BB_vec_subs[sub][t+1][next_state][σ[next_state_ind]]   # B_union_vec_subs does not have budgets at T+1 since there's no action taken there
            println("Assigned budget t = $t to t = $(t+1) for sub $sub transitioning to state $(next_state):    \n$(current_budget[sub])\n")
            current_state[sub] = next_state
        end
    end

    return budgets, actions_taken, states_visited, values_gained, costs_incurred
end