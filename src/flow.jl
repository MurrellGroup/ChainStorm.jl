const rotM = Flowfusion.Rotations(3)

schedule_f(t) = 1-(1-t)^2
const P = (FProcess(BrownianMotion(0.2f0), schedule_f), FProcess(ManifoldProcess(0.2f0), schedule_f), NoisyInterpolatingDiscreteFlow(0.2f0, 2))

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

function flowX1predictor(X0, b, model; d = identity)
    batch_dim = size(tensor(X0[1]), 4)
    f, _ = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds))
    function m(t, Xt)
        print(".")
        f, aalogits = model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames = f)
        return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
    end
    return m
end

function flowX1predictor(X0, b, model::ChainStormDesign; d = identity, pair_cond = nothing)
    batch_dim = size(tensor(X0[1]), 4)
    pair_cond = d(pair_cond)
    f, _ = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds), pair_conditioning = pair_cond)
    function m(t, Xt)
        print(".")
        f, aalogits = model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames = f, pair_conditioning = pair_cond)
        return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
    end
    return m
end

H(a; d = 2/3) = a<=d ? (a^2)/2 : d*(a - d/2)
S(a) = H(a)/H(1)

function flow_quickgen(b, model; steps = :default, d = identity, tracker = Returns(nothing))
    stps = vcat(zeros(5),S.([0.0:0.00255:0.9975;]),[0.999, 0.9998, 1.0])
    if steps isa Number
        stps = 0f0:1f0/steps:1f0
    elseif steps isa AbstractVector
        stps = steps
    end
    X0 = zero_state(b)
    X1pred = flowX1predictor(X0, b, model, d = d)
    return gen(P, X0, X1pred, Float32.(stps), tracker = tracker)
end

function flow_quickgen(b, model::ChainStormDesign; steps = :default, d = identity, tracker = Returns(nothing), pair_cond = nothing)
    stps = vcat(zeros(5),S.([0.0:0.00255:0.9975;]),[0.999, 0.9998, 1.0])
    if steps isa Number
        stps = 0f0:1f0/steps:1f0
    elseif steps isa AbstractVector
        stps = steps
    end
    X0 = zero_state(b)
    X1pred = flowX1predictor(X0, b, model, d = d, pair_cond = pair_cond)
    return gen(P, X0, X1pred, Float32.(stps), tracker = tracker)
end
