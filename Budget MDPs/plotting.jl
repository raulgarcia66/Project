using Plots

function plot_V_det(i, t::Int, B::Vector{Float64}, v::Vector{Float64}; save_plot::Bool=false, sub::Int=1)
    # B: budgets for some state and time
    # v: values for some state and time

    plot(title="V(s = $i; t = $t) deterministic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("./Budget MDPs/images/deterministic/Sub $sub V at t $t i $i det")
    end
end


function plot_V_det(i, t::Int, B::Vector{Pair{Int, Float64}}, v::Vector{Pair{Int, Float64}}; save_plot::Bool=false, sub::Int=1)
    # B: (action => budgets) for some state and time
    # v: (action => values) for some state and time

    B = map(bud -> bud.second, B)
    v = map(val -> val.second, v)
    plot_V_det(i, t, B, v; save_plot=save_plot, sub=sub)
end


function plot_V_stochastic(i, t::Int, B::Vector{Float64}, v::Vector{Float64}; save_plot::Bool=false, sub::Int=1)
    # B: budgets for state i at time t
    # v: values for state i at time t

    plot(title="V(s = $i; t = $t) stochastic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j+1]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("./Budget MDPs/images/stochastic/Sub $sub V at t $t i $i stochastic")
    end
end


function plot_V_stochastic(i, B::Vector{Float64}, v::Vector{Float64}; save_plot::Bool=false, sub::Int=1)
    # B: budgets for state i at some time
    # v: values for state i at some time

    plot(title="V(s = $i) stochastic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j+1]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("./Budget MDPs/images/stochastic/Sub $sub V at i $i stochastic")
    end
end


function plot_Q_stochastic(i, a, t, B::Vector{Float64}, v::Vector{Float64}; save_plot::Bool=false, sub::Int=1)
    # B: budgets for state i and action a at time t
    # v: values for state i and action a at time t

    plot(title="Q(s = $i; a = $a; t = $t) stochastic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j+1]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("./Budget MDPs/images/stochastic/Sub $sub Q at t $t i $i a $a stochastic")
    end
end


function plot_Q_stochastic(i, a, B::Vector{Float64}, v::Vector{Float64}; save_plot::Bool=false, sub::Int=1)
    # B: budgets for state i and action a at some time
    # v: values for state i and action a at some time

    plot(title="Q(s = $i; a = $a) stochastic", xlabel="Budget", ylims = (-0.05 * maximum(v), 1.05 * maximum(v)), legend=false)  # assuming minimum(v) = 0
    for j in 1:(length(B)-1)
        plot!([B[j]; B[j+1]], [v[j]; v[j+1]]) # ls = :dash)
        scatter!([B[j]], [v[j]])
    end
    display(scatter!([B[end]], [v[end]]))

    if save_plot
        png("./Budget MDPs/images/stochastic/Sub $sub Q at i $i a $a stochastic")
    end
end

# TODO: Create color coded function where lines correspond to actions