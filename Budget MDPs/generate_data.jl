import Random
using LinearAlgebra


# Test functions
# p = create_P_mat(num_states, num_actions, seed=1)
# r = create_r(num_states, num_actions, seed=1)

# u = create_u(num_states, num_actions, seed = 1)
# c = create_c(num_states, num_actions, seed = 1)
# r = create_r(u, c)

###############################################################################

function generate_data_rand(num_sub; seeds=-ones(num_sub), r_from_u_c=true)
    P = Array{Float64,3}[]
    C = Array{Float64,2}[]
    U = Array{Float64,2}[]
    R = Array{Float64,2}[]
    # U_terminal = create_u(num_states, num_actions, seed = seed)
    # U_terminal = zeros(num_states)  # reward when no action is taken (e.g., at end of planning horizon); might not be used

    if seeds == -ones(num_sub)
        seeds = 1:num_sub
    end

    for seed in seeds
        push!(P, create_P_mat(num_states, num_actions, seed = seed))

        if r_from_u_c
            ## Create r from u and c
            u = create_u(num_states, num_actions, seed = seed)
            push!(U, u)
            c = create_c(num_states, num_actions, seed = seed)
            push!(C, c)
            push!(R, create_r(u, c))
        else
            ## Create r independently
            # push!(r, create_r(num_states, num_actions, seed = seed))
        end
    end

    return P, C, U, R
end

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

    u_col = rand(5:10, num_states)   # Want u >= c most entries
    u = copy(u_col)
    for _ = 1:(num_actions-1)
        u = hcat(u, u_col)
    end
    u[:, end] .= 0   # for the free action

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

    return c
end
