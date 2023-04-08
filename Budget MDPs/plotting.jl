using Plots

function plot_V(i, t, B::Vector{Float64}, v::Vector{Float64}; save_plot=false)
    # B: budgets for some state and time
    # v: values for some state and time

    plot(title="V(s = $i; t = $t)", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for i in 1:(length(B)-1)
        plot!([B[i]; B[i+1]], [v[i]; v[i]]) # ls = :dash)
        scatter!([B[i]], [v[i]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("V at i $i t $t")
    end
end


# # TODO: Create color coded function where lines correspond to actions
# # Difficulty lies in getting the axes to just show unique colors for the lines
# function plot_V(i, t, B::Vector{Pair{T,Float64}}, v::Vector{Pair{T,Float64}}; save_plot=false) where {T}
#     # B: budgets for some state and time as (action => budget) pairs
#     # v: values for some state and time as (action => value) pairs

#     plot(title="V(s = $i; t = $t)", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
#     for i in 1:(length(B)-1)
#         plot!([B[i]; B[i+1]], [v[i]; v[i]]) # ls = :dash)
#         scatter!([B[i]], [v[i]])
#     end
#     display(scatter!([B[end]], [v[end]]))

#     if save_plot
#         png("V at i $i t $t")
#     end
# end