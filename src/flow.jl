const rotM = Flowfusion.Rotations(3)

schedule_f(t) = 1-(1-t)^2
const P = (FProcess(BrownianMotion(0.2f0), schedule_f), FProcess(ManifoldProcess(0.2f0), schedule_f), NoisyInterpolatingDiscreteFlow(0.2f0, K = 2, dummy_token = 21))

#Bringing Alexander's version in - this should be replaced by the "full" solution:
function rev_NoisyInterpolatingDiscreteFlow(noise; K = 1, dummy_token::T = nothing) where T
    if (K > 1 && isnothing(dummy_token)) 
        @warn "NoisyInterpolatingDiscreteFlow: If K>1 things might break if your X0 is not the `dummy_token` (which should also be passed to NoisyInterpolatingDiscreteFlow)."
    end
    return NoisyInterpolatingDiscreteFlow{T}(
                t -> oftype(t,1-(1-cos((π/2)*(1-t)))^K), #K1
                t -> oftype(t,(noise * sin(π*t))), #K2
                t -> oftype(t,(K * (π/2) * cos((π/2) * t) * (1 - sin((π/2) * t))^(K - 1))), #dK1
                t -> oftype(t,(noise*π*cos(π*t))), #dK2
                dummy_token
                )
end

function reverse_process(P)
    continuous_schedule =   t -> 1 - P[1].F(1 - t)
    manifold_schedule =   t -> 1 - P[2].F(1 - t)
    κ₁ =  t -> 1 - P[3].κ₁(1 - t)
    dκ₁ = t ->     P[3].dκ₁(1 - t)
    κ₂ =  t -> 1 - P[3].κ₂(1 - t)
    dκ₂ = t ->     P[3].dκ₁(1 - t)
    #This is a hack fix:
    #(Flowfusion.FProcess(P[1].P, continuous_schedule), Flowfusion.FProcess(P[2].P, manifold_schedule), Flowfusion.NoisyInterpolatingDiscreteFlow(κ₁, dκ₁, κ₂, dκ₂, P[3].mask_token))
    (Flowfusion.FProcess(P[1].P, continuous_schedule), Flowfusion.FProcess(P[2].P, manifold_schedule), rev_NoisyInterpolatingDiscreteFlow(0.2f0, K = 2, dummy_token = 21))
end

function compound_state(b)
    L,B = size(b.aas)
    cmask = b.aas .< 100
    X1locs = MaskedState(ContinuousState(b.locs), cmask, b.padmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(b.rots, dims=(3,4))), cmask, b.padmask)
    X1aas = MaskedState(DiscreteState(21, Flux.onehotbatch(b.aas, 1:21)), cmask, b.padmask)
    return (X1locs, X1rots, X1aas)
end

function zero_state(b)
    L,B = size(b.aas)
    cmask = b.aas .< 100
    X0locs = MaskedState(ContinuousState(randn(Float32, size(b.locs))), cmask, b.padmask)
    X0rots = MaskedState(ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, L*B)), L, B)), cmask, b.padmask)
    X0aas = MaskedState(DiscreteState(21, Flux.onehotbatch(similar(b.aas) .= 21, 1:21)), cmask, b.padmask)
    return (X0locs, X0rots, X0aas)
end

function training_sample(b)
    X0 = zero_state(b)
    X1 = compound_state(b)
    t = rand(Float32, 1, size(b.aas,2))
    Xt = bridge(P, X0, X1, t)
    rotξ = Guide(Xt[2], X1[2])
    return (; t, Xt, X1, rotξ, chainids = b.chainids, resinds = b.resinds)
end

function losses(hatframes, aalogits, ts)
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hatframes)), tensor(ts.Xt[2]))
    hatloc, hatrot, hataas = (values(translation(hatframes)), rotangent, aalogits)
    l_loc = floss(P[1], hatloc, ts.X1[1], scalefloss(P[1], ts.t, 2, 0.2f0)) / 2
    l_rot = floss(P[2], hatrot, ts.rotξ, scalefloss(P[2], ts.t, 2, 0.2f0)) / 10
    l_aas = floss(P[3], hataas, ts.X1[3], scalefloss(P[3], ts.t, 1, 0.2f0)) / 100
    return l_loc, l_rot, l_aas
end

function flowX1predictor(X0, b, model; d = identity, smooth = 0)
    batch_dim = size(tensor(X0[1]), 4)
    f, _ = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds))
    prev_trans = values(translation(f))
    T = eltype(prev_trans)
    function m(t, Xt)
        f, aalogits = model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames = f)
        values(translation(f)) .= prev_trans .* T(smooth) .+ values(translation(f)) .* T(1-smooth)
        prev_trans = values(translation(f))
        return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
    end
    return m
end

function flowX0predictor(X0, b, model, P; d = identity, smooth = 0) # Forces P to be a FProcess and doesn't work for some reason for P Deterministic
    batch_dim = size(tensor(X0[1]), 4)
    ff, _ = model(d(ones(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds)) # ones makes it start at time = 1
    if P[1].P isa Deterministic
        v = 0
    else
        v = P[1].P.v
    end
    function m(rt, Xt)
        ff, aalogits = model(d(1-rt .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames=ff)
        aalogits = deepcopy(cpu(aalogits))
        X1Hat = deepcopy(cpu(ff))
        t = 1f0 .- P[1].F.(rt .+ zeros(Float32, 1, batch_dim))
        t[t .>= 0.999] .= 0.999
        values(translation(X1Hat)) .= (tensor(Xt[1]) .- values(translation(X1Hat)) .* t) ./ (1 .- t + v .* t)
        M = Xt[2].S.M
        p = eachslice(tensor(Xt[2]), dims=(3, 4))
        tangent = -t ./ (1 .- t) .* log.((M,), p, eachslice(values(linear(X1Hat)), dims=(3, 4)))
        X0Hat = exp.((M,), p, tangent)
        values(linear(X1Hat)) .= stack(X0Hat)
        T = eltype(aalogits)
        aalogits .= T(-Inf)
        aalogits[21,:,:] .= 0
        return (cpu(values(translation(X1Hat))), ManifoldState(rotM, eachslice(cpu(values(linear(X1Hat))), dims=(3,4))), cpu(softmax(aalogits))), (cpu(values(translation(ff))), ManifoldState(rotM, eachslice(cpu(values(linear(ff))), dims=(3,4))), cpu(softmax(aalogits)))
    end
    return m
end

function bind_flowX1predictor(X0, b, model, recorded; d = identity, smooth = 0, meanshift = true)
    recdim = size(tensor(recorded[end][3][1]), 3)
    batch_dim = size(tensor(X0[1]), 4)
    f, _ = cpu(model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds)))
    values(translation(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][1]) # Might be more sensible to do a weighted average of X̂₁ and (1-t)*(Xₜ₊Δₜ - Xₜ)/Δt
    values(linear(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][2])
    f, _ = cpu(model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds), sc_frames=d(f)))
    recmean = Flux.mean(values(translation(f))[:, :, 1:recdim, :], dims = 3)
    forcemean = Flux.mean(tensor(recorded[end][3][1]), dims = 3)
    values(translation(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][1])
    if meanshift
        values(translation(f))[:, :, 1:recdim, :] .+= forcemean .- recmean
    end
    values(linear(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][2])
    function m(t, Xt)
        f, aalogits = cpu(model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames = d(f)))
        recmean = Flux.mean(values(translation(f))[:, :, 1:recdim, :], dims = 3)
        forcemean = Flux.mean(tensor(recorded[end][3][1]), dims = 3)
        values(translation(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][1])
        if meanshift #This is to shift the binder over by the amount the target would have shifted, in the other direction:
            values(translation(f))[:, :, recdim+1:end, :] .+= forcemean .- recmean
        end
        values(linear(f))[:, :, 1:recdim, :] .= tensor(recorded[end][3][2])
        return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
    end
    return m
end

H(a; d = 2/3) = a<=d ? (a^2)/2 : d*(a - d/2)
S(a) = H(a)/H(1)

function flow_quickgen(P, b, X0, model; is_reverse = false, steps = :default, d = identity, tracker = Returns(nothing), smooth = 0.6, record = [], progress_bar=identity, snap_time = 0)
    stps = vcat(zeros(5),S.([0.0:0.00255:0.9975;]),[0.999, 0.9998, 1.0])

    if steps isa Number 
        stps = 0f0:1f0/steps:1f0
    elseif steps isa AbstractVector
        stps = steps
    end
    b.locs .= tensor(X0[1])
    #b.aas .= convert(Matrix{Int64}, tensor(X0[3]).indices)
    b.aas .= unhot(X0[3]).S.state
    if !is_reverse && length(record) == 0
        X1pred = flowX1predictor(X0, b, model, d = d, smooth = smooth)
        return gen(P, X0, X1pred, Float32.(stps), tracker = tracker, progress_bar = progress_bar)
    elseif is_reverse
        X0pred = flowX0predictor(X0, b, model, P, d = d, smooth = smooth)
        return reverse_gen(P, X0, X0pred, Float32.(1 .- reverse(stps)), record, tracker = tracker, progress_bar = progress_bar, snap_time = snap_time), record
    else
        X1pred = bind_flowX1predictor(X0, b, model, record, d = d, smooth = smooth)
        return bind_gen(P, X0, X1pred, Float32.(stps), record, tracker = tracker, progress_bar = progress_bar)            
    end
end
