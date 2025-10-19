using Pkg
Pkg.activate("reversal", shared=true)
#Pkg.develop(path=".")
#Pkg.add(["CUDA", "cuDNN", "Flux"])

using ChainStorm, ChainStorm.Flowfusion, ChainStorm.ProteinChains
using CUDA, Flux

model = load_model() |> gpu;

struc = pdb"7RBY"1[1:1];
target_length = length(struc[1]);
batch_target = ChainStorm.pdb2batch(struc);

@time recorded, rev_g = flow_quickgen(
    ChainStorm.P, batch_target, ChainStorm.compound_state(batch_target), model;
    is_reverse = true, d = gpu, steps = 0f0:0.001f0:1f0, snap_time = 0.9f0);

#=new_lengths = [122, 114]
batch = dummy_batch([ChainStorm.lengths_from_chainids(batch_target.chainids); new_lengths])
batch.resinds .= [batch_target.resinds; 1:new_lengths[1]; 1:new_lengths[2]]
X₀ = ChainStorm.zero_state(batch)=#

tracker = ChainStorm.Tracker()
@time fwd_g = flow_quickgen(
    ChainStorm.P, batch_target, rev_g, model;
    tracker, d = gpu, steps = 0f0:0.001f0:1f0);

id = join(ChainStorm.lengths_from_chainids(batch_target.chainids),'_')*"-"*join(rand('A':'Z', 4))

export_pdb("$(id)_bind.pdb", fwd_g, batch_target.chainids, batch_target.resinds) #<- Save PDB
