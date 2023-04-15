using Plots

function plot_V_det(i, t, B::Vector{Float64}, v::Vector{Float64}; save_plot=false)
    # B: budgets for some state and time
    # v: values for some state and time
    # TODO: Add action ticks to scatter to denote the action

    plot(title="V(s = $i; t = $t) deterministic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("V at i $i t $t det")
    end
end


# TODO: Fix this method for when B and v are B_union and v_union
function plot_V_det(i, t, B::Vector{Pair{Int, Float64}}, v::Vector{Pair{Int, Float64}}; save_plot=false)
    # B: budgets for some state and time
    # v: values for some state and time
    # TODO: Add action ticks to scatter to denote the action

    plot(title="V(s = $i; t = $t) deterministic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("V at i $i t $t det")
    end
end


function plot_V_stochastic(i, t, B::Vector{Float64}, v::Vector{Float64}; save_plot=false)
    # B: budgets for some state and time
    # v: values for some state and time
    # TODO: Add action ticks to scatter to denote the action

    plot(title="V(s = $i; t = $t) stochastic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j+1]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("V at i $i t $t stochastic")
    end
end


# # TODO: Create color coded function where lines correspond to actions