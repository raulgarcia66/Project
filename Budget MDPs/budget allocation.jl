using JuMP
using Gurobi

"""
Solve useful budget assignment problem (UBAP).
"""
function solve_UBAP(num_subs::Int, init_state_subs::Vector{T}, global_budget::W, B_vec_subs::Vector{Vector{Vector{Vector{Float64}}}}, v_vec_subs::Vector{Vector{Vector{Vector{Float64}}}}) where {T, W <: Real}
    M = [i for i in 1:num_subs]
    L = [1:length(B_vec_subs[i][1][init_state_subs[i]]) for i = 1:num_subs]
    # L = [collect(1:length(BB_vec_subs[i][1][init_state_subs[i]])) for i = 1:num_subs]
    # @variable(model, x[i in 1:num_subs, k in 1:length(BB_vec_subs[i][1][init_state_subs[i]])], Bin)

    model = Model(Gurobi.Optimizer)
    @variable(model, x[i in M, k in L[i]], Bin)

    @constraint(model, sum([sum([B_vec_subs[i][1][init_state_subs[i]][k] * x[i,k] for k in L[i]]) for i in M]) <= global_budget)
    @constraint(model, [i in M] , sum([x[i,k] for k in L[i]]) <= 1)

    sum([sum([B_vec_subs[i][1][init_state_subs[i]][k] * x[i,k] for k in L[i]]) for i in M])

    @objective(model, Max, sum([sum([v_vec_subs[i][1][init_state_subs[i]][k] * x[i,k] for k in L[i]]) for i in M]))

    # println("$model")
    optimize!(model)

    return value.(x), objective_value(model)
end


function extract_budget_index(x_vals, num_subs::Int)
    indices = Vector{Int}(undef, num_subs)
    for i in 1:num_subs
        for k in eachindex(x_vals[i,:])
            if x_vals[i,k[1]] > 0.5
                indices[i] = k[1]   # k is a one-element tuple
                break
            end
        end
    end
    return indices
end