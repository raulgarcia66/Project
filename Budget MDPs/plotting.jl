using Plots

function plot_V(i, t, B, v)
    # B: (action => budget) vector
    # v: (action => value) vector

    B = map(b -> b.second, B)
    v = map(val -> val.second, v)

    plot(title="V(s = $i; t = $t)", xlabel="Budget", legend=false)
    for i in 1:(length(B)-1)
        plot!([B[i]; B[i+1]], [v[i]; v[i]], ls = :dash)
        scatter!([B[i]], [v[i]])
    end
    display(scatter!([B[end]], [v[end]]))
end