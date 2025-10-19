using Pkg
Pkg.activate("reversal", shared=true)

using ChainStorm, ChainStorm.Flowfusion, ChainStorm.ProteinChains
using CUDA, Flux, Random

model = load_model() |> gpu

struc = pdb"7RBY"1[1:1]
batch_target = ChainStorm.pdb2batch(struc)
X_target = ChainStorm.compound_state(batch_target)

phase_points_rev = Float32[1.0, 0.0]
@time _, reverse_records, _ = ChainStorm.flex_gen(
    ChainStorm.P, batch_target, X_target, model, phase_points_rev;
    step_size = -0.005f0,
    record_indices = [1],
    d = gpu,
    snap_time = 0.9f0,
)
recorded = get(reverse_records, 1, nothing)
recorded === nothing && error("Reverse phase did not produce a recording")
recorded = deepcopy(recorded)

new_lengths = [122, 114]
batch = dummy_batch([ChainStorm.lengths_from_chainids(batch_target.chainids); new_lengths])
batch.resinds .= [batch_target.resinds; 1:new_lengths[1]; 1:new_lengths[2]]
X₀ = ChainStorm.zero_state(batch)

tracker = ChainStorm.Tracker()
phase_points_fw = Float32[0.0, 1.0]
initial_record = (_, _) -> deepcopy(recorded)
@time fwd_state, _, _ = ChainStorm.flex_gen(
    ChainStorm.P, batch, X₀, model, phase_points_fw;
    step_size = 0.005f0,
    record_indices = [1],
    initial_recorded = initial_record,
    tracker = tracker,
    d = gpu,
)

id = join(ChainStorm.lengths_from_chainids(batch.chainids), '_') * "-" * String(rand('A':'Z', 4))

export_pdb("$(id)_bind.pdb", fwd_state, batch.chainids, batch.resinds)
