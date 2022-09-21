# KI2022 UDE presentation

# 1. introduction to julia

```julia
function fibonacci(n)
    n in (0, 1) && return n
    return fibonacci(n-1) + fibonacci(n-2)
end
fibonacci(40)
```

```julia
function fibonacci_improved(n)
    array = fill(-1, n+1)
    array[0+1] = 0
    array[1+1] = 1
    return fibonacci_improved(n, array)
end
function fibonacci_improved(n, array)
    if array[n+1] == -1
        array[n+1] = fibonacci_improved(n-1, array) + fibonacci_improved(n-2, array)
    end
    return array[n+1]
end
```

# 2. introduction to deep learning

```julia
function generate_data(rng::Random.AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end

(x, y) = generate_data(rng)

Plots.plot(x -> evalpoly(x, (0, -2, 1)), x[1, :]; label=false)
Plots.scatter!(x[1, :], y[1, :]; label=false, markersize=3)
```

```julia
function loss_function(x, y, ps, st)
    y_pred, st = Lux.apply(model, x, ps, st)
    sum(abs2, y .- y_pred), st
end
```

# 3. deep-dive into UDE

```julia
Plots.plot(predict(p_initial))
```

```julia
loss(p_initial), shooting_loss(p_initial)
```

```julia
losses = Float32[]
minimizer = p_initial

for (opt_alg, maxiters, loss_func) = [
        (OptimizationOptimisers.Adam(0.01), 200, shooting_loss)
        (OptimizationOptimisers.Adam(0.01), 1_000, loss)
    ]
    opt_func = Optimization.OptimizationFunction((ps, _) -> loss_func(ps), Optimization.AutoZygote())   
    opt_prob = Optimization.OptimizationProblem(opt_func, minimizer) 
    opt_sol = solve(opt_prob, opt_alg, maxiters = maxiters, callback = callback)
    minimizer = opt_sol.minimizer
end
ps_trained = minimizer
```

```julia
# Look at long term prediction
t_long = (0.0f0, 50.0f0)
estimate_long = solve(ode_prob_nn, saveat = 0.25f0, tspan = t_long, p = ps_trained)
Plots.plot(estimate_long.t, transpose(xscale .* Array(estimate_long)), xlabel = "t", ylabel = "x(t),y(t)")
```

```julia
ideal_problem = DataDrivenDiffEq.DataDrivenProblem(X_pred, Y=Y_ideal)
ideal_res = solve(ideal_problem, basis, opt, maxiter = 10_000, progress = true, normalize = false, denoise = true)
println(DataDrivenDiffEq.result(ideal_res))
println(DataDrivenDiffEq.parameter_map(ideal_res))
```

Extra code which was cut out to save time

```julia
DX_ddd_fullreal = ddd_sol_fullnoisy(ddd_prob_fullreal.X, ddd_sol_fullreal.parameters, t)
p1 = Plots.plot(t, DX[1,:], label="true", ylabel="du1", xlabel="t")
Plots.plot!(t, ddd_prob_fullnoisy.DX[1,:], label="collocation approximation")
Plots.plot!(t, DX_ddd_fullreal[1,:], label="global symbolic regression")

p2 = Plots.plot(t, DX[2,:], label="true", ylabel="du2", xlabel="t")
Plots.plot!(t, ddd_prob_fullnoisy.DX[2,:], label="collocation approximation")
Plots.plot!(t, DX_ddd_fullreal[2,:], label="global symbolic regression")

Plots.plot(p1, p2, layout=(2,1))
```

```julia
# standard Lotka Volterra 
Y_ideal = [
    -ps_ideal[2] * (X_ideal[1,:] .* X_ideal[2,:])'
    ps_ideal[4] * (X_ideal[1,:] .* X_ideal[2,:])'
]

# prediction of global data driven approach, minus linear learned terms
Y_ddd_fullreal = DX_ddd_fullreal - [ps_ideal[1], -ps_ideal[3]] .* ddd_prob_fullreal.X

# Neural network guess
Y_nn, _st_lux = model_lux(X_pred, ps_trained.ps_lux, st_lux)

Y_ddd_nn = ddd_sol_nn(X_pred, ddd_sol_nn.parameters)
p1 = Plots.plot(Y_ddd_nn[1,:], label="symbolic regression ude", ylabel="interactive part du1")
Plots.plot!(Y_nn[1,:], label="ude prediction")
Plots.plot!(Y_ideal[1,:], label="true")
Plots.plot!(Y_ddd_fullreal[1,:], label="symbolic regression global")

p2 = Plots.plot(Y_ddd_nn[2,:], label="symbolic regression ude", ylabel="interactive part du2")
Plots.plot!(Y_nn[2,:], label="ude prediction")
Plots.plot!(Y_ideal[2,:], label="true")
Plots.plot!(Y_ddd_fullreal[2,:], label="symbolic regression global")

Plots.plot(p1, p2, layout=(2,1))
```

# 4. introduction to bayesian differential equations

```julia
Plots.plot(chain)
```
