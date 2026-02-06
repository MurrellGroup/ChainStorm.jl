module ChainStorm

using Flowfusion, ForwardBackward, Flux, RandomFeatureMaps, Onion, InvariantPointAttention, BatchedTransformations, ProteinChains, DLProteinFormats, HuggingFaceApi, JLD2
using OneHotArrays #Added

include("flow.jl")
include("model.jl")

function load_model(; checkpoint = "ChainStormV1.jld2")
    file = hf_hub_download("MurrellLab/ChainStorm", checkpoint)
    return Flux.loadmodel!(ChainStormV1(), JLD2.load(file, "model_state"))
end

chainids_from_lengths(lengths) = vcat([repeat([i],l) for (i,l) in enumerate(lengths)]...)
function gen2prot(samp, chainids, resnums; name = "Gen", )
    d = Dict(zip(0:25,'A':'Z'))
    chain_letters = get.((d,), chainids, 'Z')
    ProteinStructure(name, Atom{eltype(tensor(samp[1]))}[], DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), chain_letters, resnums)[1])
 end
export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

function first_trajectory(paths)
    ts = paths.t
    xt_locs = [tensor(x[1])[:,1,:,1] for x in paths.xt]
    xt_rots = [tensor(x[2])[:,:,:,1] for x in paths.xt]
    xt_aas = [tensor(ChainStorm.unhot(x[3]))[:,1] for x in paths.xt]
    x̂1_locs = [tensor(x[1])[:,1,:,1] for x in paths.x̂1]
    x̂1_rots = [tensor(x[2])[:,:,:,1] for x in paths.x̂1]
    x̂1_aas = [tensor(x[3])[:,:,1] for x in paths.x̂1]
    trajectory = (;ts, xt_locs,xt_rots,xt_aas,x̂1_locs,x̂1_rots,x̂1_aas)
    return trajectory
end

function dummy_batch(chain_lengths)
    chainid_vec = chainids_from_lengths(chain_lengths)
    L = length(chainid_vec)
    chainids = reshape(chainid_vec, :, 1)
    resinds = similar(chainids)[:] .= 1:L
    padmask = trues(L, 1)
    aas = 21*ones(Int, L, 1)
    locs = randn(Float32, 3, 1, L, 1)
    return (;chainids, resinds, padmask, aas, locs)
end

export training_sample, P, ChainStormV1, losses, flow_quickgen, export_pdb, gen2prot, dummy_batch, first_trajectory, load_model

end
