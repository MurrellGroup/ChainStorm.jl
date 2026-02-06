const rotM = Flowfusion.Rotations(3)

schedule_f(t) = 1-(1-t)^2
#const P = (FProcess(BrownianMotion(0.2f0), schedule_f), FProcess(ManifoldProcess(0.2f0), schedule_f), NoisyInterpolatingDiscreteFlow(0.2f0, K = 2, dummy_token = 21))

P = (FProcess(BrownianMotion(0.2f0), schedule_f), FProcess(ManifoldProcess(0.2f0), schedule_f))

function compound_state(b)
    L,B = size(b.aas)
    cmask = b.aas .< 100
    X1locs = MaskedState(ContinuousState(b.locs), cmask, b.padmask)
    X1rots = MaskedState(ManifoldState(rotM,eachslice(b.rots, dims=(3,4))), cmask, b.padmask)
    #X1aas = MaskedState(DiscreteState(21, Flux.onehotbatch(b.aas, 1:21)), cmask, b.padmask)
    #return (X1locs, X1rots, X1aas)
    return (X1locs, X1rots)
end

#Added seq_to_masked_state 
function seq_to_masked_state(seq_int; K=21)
    N = length(seq_int)
    B = 1  # batch size
    
    # Step 1: Create 2D one-hot to get the indices
    oh_2d = Flux.onehotbatch(seq_int, 1:K)  # K×N OneHotMatrix
    
    # Step 2: Reshape indices to 3D (N×B matrix) for batch dimension
    indices_3d = reshape(oh_2d.indices, N, B)  # N×B Matrix{UInt32}
    
    # Step 3: Create 3D OneHotArray (K×N×B)
    oh_3d = OneHotArrays.OneHotArray(indices_3d, K)
    
    # Step 4: Create DiscreteState with 3D one-hot array
    ds = ForwardBackward.DiscreteState(K, oh_3d)
    
    # Step 5: Create masks (N×B)
    cmask = trues(N, B)  # conditioning mask
    lmask = trues(N, B)  # length/padding mask
    
    # Step 6: Create MaskedState
    return Flowfusion.MaskedState(ds, cmask, lmask)
end


function zero_state(b)
    L,B = size(b.aas)
    cmask = b.aas .< 100
    X0locs = MaskedState(ContinuousState(randn(Float32, size(b.locs))), cmask, b.padmask)
    X0rots = MaskedState(ManifoldState(rotM, reshape(Array{Float32}.(Flowfusion.rand(rotM, L*B)), L, B)), cmask, b.padmask)
    #X0aas = MaskedState(DiscreteState(21, Flux.onehotbatch(similar(b.aas) .= 21, 1:21)), cmask, b.padmask)
    #return (X0locs, X0rots, X0aas)
    return (X0locs, X0rots)
end

function training_sample(b)
    X0 = zero_state(b)
    X1 = compound_state(b)
    t = rand(Float32, 1, size(b.aas,2))
    Xt = bridge(P, X0, X1, t)
    rotξ = Guide(Xt[2], X1[2])
    #return (; t, Xt, X1, rotξ, chainids = b.chainids, resinds = b.resinds)
    return (; t, Xt, X1, rotξ, aas = b.aas, chainids = b.chainids, resinds = b.resinds)
end

#function losses(hatframes, aalogits, ts)
function losses(hatframes, ts)
    rotangent = Flowfusion.so3_tangent_coordinates_stack(values(linear(hatframes)), tensor(ts.Xt[2]))
    #hatloc, hatrot, hataas = (values(translation(hatframes)), rotangent, aalogits)
    hatloc, hatrot = (values(translation(hatframes)), rotangent)
    l_loc = floss(P[1], hatloc, ts.X1[1], scalefloss(P[1], ts.t, 2, 0.2f0)) / 2
    l_rot = floss(P[2], hatrot, ts.rotξ, scalefloss(P[2], ts.t, 2, 0.2f0)) / 10
    #l_aas = floss(P[3], hataas, ts.X1[3], scalefloss(P[3], ts.t, 1, 0.2f0)) / 100
    #return l_loc, l_rot, l_aas
    return l_loc, l_rot
end

function flowX1predictor(X0, b, model; d = identity, smooth = 0)
    batch_dim = size(tensor(X0[1]), 4)
    #f, aalogtis = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds))
    f = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.aas), d(b.chainids), d(b.resinds))
    prev_trans = values(translation(f))
    T = eltype(prev_trans)
    function m(t, Xt)
        print(".")
        #f, aalogits = model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.chainids), d(b.resinds), sc_frames = f) 
        f = model(d(t .+ zeros(Float32, 1, batch_dim)), d(Xt), d(b.aas), d(b.chainids), d(b.resinds), sc_frames = f) 
        values(translation(f)) .= prev_trans .* T(smooth) .+ values(translation(f)) .* T(1-smooth)
        prev_trans = values(translation(f))
        #return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
        return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4)))
    end
    return m
end

H(a; d = 2/3) = a<=d ? (a^2)/2 : d*(a - d/2)
S(a) = H(a)/H(1)

function flow_quickgen(b, model; steps = :default, d = identity, tracker = Returns(nothing), smooth = 0.6)
    stps = vcat(zeros(5),S.([0.0:0.00255:0.9975;]),[0.999, 0.9998, 1.0])
    if steps isa Number
        stps = 0f0:1f0/steps:1f0
    elseif steps isa AbstractVector
        stps = steps
    end
    X0 = zero_state(b)
    X1pred = flowX1predictor(X0, b, model, d = d, smooth = smooth)
    return gen(P, X0, X1pred, Float32.(stps), tracker = tracker)
end
