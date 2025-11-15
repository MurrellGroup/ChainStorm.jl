using Pkg
Pkg.activate("reversal", shared=true)
#Pkg.develop(path=".")
#Pkg.add(["CUDA", "cuDNN", "Flux"])

using ChainStorm, Flowfusion, ChainStorm.ProteinChains
using CUDA, Flux

model = load_model() |> gpu;

struc = pdb"7RBY"1[1:1];
batch_target = ChainStorm.pdb2batch(struc);
X0 = compound_state(batch_target)

phases = [
    Phase(1.0 => 0.0),
    Phase(0.0 => 1.0, use_record=true, new_lengths=[100, 120]),
    Phase(1.0 => 0.5, record_dim=195),
    Phase(0.5 => 1.0, use_record=true),
    Phase(1.0 => 0.75, record_dim=195),
    Phase(0.75 => 1.0, use_record=true),
]

out = flex_quickgen(P, batch_target, X0, model; phases, d=gpu);

dir = "movie"
isdir(dir) || mkdir(dir)
frame_index = 0; for (i, (phase, (X1, b, tracker))) in enumerate(zip(phases, out))
    for (j, Xₜ) in enumerate(tracker.xt)
        frame_index += 1
        export_pdb(
            start, stop = phase.interval,
            "$dir/$frame_index-$i-$j-$start-$stop.pdb",
            Xₜ, b.chainids, b.resinds)
    end
end
