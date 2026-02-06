#In addition to ChainStorm, also install these:
using Pkg
Pkg.add(["JLD2", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "DLProteinFormats"])
Pkg.add(["CUDA", "cuDNN"])

using ChainStorm, DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA
device = gpu

dat = load(PDBSimpleFlat);

model = ChainStormV1(384, 3, 3) |> device
sched = burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model)

for epoch in 1:100
    batchinds = sample_batched_inds(dat,l2b = length2batch(1500, 1.9))
    for (i, b) in enumerate(batchinds)
        bat = batch_flatrecs(dat[b])
        ts = training_sample(bat) |> device
        sc_frames = nothing
        if epoch > 1 && rand() < 0.5
            sc_frames, _ = model(ts.t, ts.Xt, ts.chainids, ts.resinds)
        end
        l, grad = Flux.withgradient(model) do m
            fr, aalogs = m(ts.t, ts.Xt, ts.chainids, ts.resinds, sc_frames = sc_frames)
            l_loc, l_rot, l_aas = losses(fr, aalogs, ts)
            l_loc + l_rot + l_aas
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        println(l)
    end
    jldsave("model_epoch_$epoch.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end