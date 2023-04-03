using Plots

function plot_V(i, t, B, v; save_plot=false)
    # B: useful budgets
    # v: values

    println("B: $B \nv = $v\n")
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