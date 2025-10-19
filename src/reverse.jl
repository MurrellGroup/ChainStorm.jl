using Flowfusion:
    resolveprediction, mask, step,
    tensor, UProcess, UState, unhot

function reverse_gen(
    P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}},
    model, steps::AbstractVector, record;
    #recdim = Flowfusion.lastsize(X₀, offset=1),
    tracker = Returns(nothing),
    midpoint = false,
    snap_time = 0,
)
    Xₜ = copy.(X₀)
    push!(record, (1, X₀, nothing))
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        T = eltype(s₁)
        s₁ = s₁ == s₂ ? s₁ - T(0.001) : s₁
        t = midpoint ? (s₁ + s₂) / 2 : s₁
        X̂₀, X̂₁ = model(t, Xₜ)
        X̂₀ = resolveprediction(X̂₀, Xₜ)
        X̂₁ = resolveprediction(X̂₁, Xₜ)
        Xₜ = mask(step(P, Xₜ, X̂₀, s₁, s₂), X₀)
        if t < snap_time
            fakeX̂₁ = deepcopy(X̂₁)
            tensor(fakeX̂₁[1]) .= tensor(X₀[1])
            tensor(fakeX̂₁[2]) .= tensor(X₀[2])
            push!(record, (
                1-s₂,
                deepcopy(Xₜ),# 1:recdim, offset=1),
                fakeX̂₁#, 1:recdim, offset=1)
            ))
        else
            push!(record, (
                1-s₂,
                deepcopy(Xₜ),# 1:recdim, offset=1),
                deepcopy(X̂₁)#, 1:recdim, offset=1)
            ))
        end
        tracker(1-t, Xₜ, X̂₁)
    end
    return Xₜ
end

function bind_gen(
    P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}},
    model, steps::AbstractVector, record;
    tracker = Returns(nothing),
    midpoint = false,
)
    Xₜ = copy.(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : s₁
        X̂₁ = resolveprediction(model(t, Xₜ), Xₜ)
        #Changes xt 
        s₁, old_Xₜ, _ = record[end]
        pop!(record)
        old_size = size(tensor(old_Xₜ[1]), 3)
        tensor(Xₜ[1])[:, :, 1:old_size, :] .= tensor(old_Xₜ[1])
        tensor(Xₜ[2])[:, :, 1:old_size, :] .= tensor(old_Xₜ[2])
        tensor(Xₜ[3]).indices[1:old_size, :] .= tensor(old_Xₜ[3]).indices
        Xₜ = mask(step(P, Xₜ, X̂₁, s₁, s₂), X₀)
        if length(record) == 1
            tensor(Xₜ[1])[:, :, 1:old_size, :] .= tensor(old_Xₜ[1])
            tensor(Xₜ[2])[:, :, 1:old_size, :] .= tensor(old_Xₜ[2])
            tensor(Xₜ[3]).indices[1:old_size, :] .= tensor(old_Xₜ[3]).indices
        end
        tracker(t, Xₜ, X̂₁)
    end
    return Xₜ
end
