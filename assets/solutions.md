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
p1 = Plots.plot(predict(ps_initial), ylim = (0,6))
p2 = Plots.plot(predict(ps_trained), ylim = (0,6))
Plots.plot(p1, p2, layout=(3,1))
```

```julia
Plots.plot(predict(p_initial))
```

```julia
loss(p_initial)
```

```julia
shooting_loss(p_initial)
```

```julia
minimizer = p_initial

for (opt_alg, maxiters, loss_func) = [
        (OptimizationOptimisers.Adam(0.1), 100, shooting_loss),
        (OptimizationOptimisers.Adam(0.01), 2_000, loss),
    ]
    opt_func = Optimization.OptimizationFunction((ps, _) -> loss_func(ps), Optimization.AutoZygote())   
    opt_prob = Optimization.OptimizationProblem(opt_func, minimizer) 
    opt_sol = solve(opt_prob, opt_alg, maxiters = maxiters, callback = callback)
    minimizer = opt_sol.minimizer
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
end
p_trained = minimizer
```

```julia
# Look at long term prediction
t_long = (0.0f0, 50.0f0)
estimate_long = solve(ode_prob_nn, saveat = 0.25f0, tspan = t_long, p = p_trained)
Plots.plot(estimate_long.t, transpose(xscale .* Array(estimate_long)), xlabel = "t", ylabel = "x(t),y(t)")
```

```julia
ideal_problem = DataDrivenDiffEq.DataDrivenProblem(X_pred, Y=Y_ideal)
ideal_res = solve(ideal_problem, basis, opt, maxiter = 10_000, progress = true, normalize = false, denoise = true)
println(DataDrivenDiffEq.result(ideal_res))
println(DataDrivenDiffEq.parameter_map(ideal_res))
```

# 4. introduction to bayesian differential equations

```julia
Plots.plot(chain)
```
