#A minimal training script:
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.activate(".")

Pkg.add(["JLD2", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "DLProteinFormats"])
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
Pkg.develop(path="../")

using ChainStorm, DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch

using CUDA        #<- If GPU
device = gpu      #<- If GPU
#device = identity #<- if no GPU

dat = load(PDBSimpleFlat);

model = FlowcoderSC(384, 6, 6) |> device
sched = burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model)

for epoch in 1:100
    batchinds = sample_batched_inds(dat,l2b = length2batch(1250, 1.9)) #<- For a 48gb GPU
    for (i, b) in enumerate(batchinds)
        bat = batch_flatrecs(dat[b])
        ts = training_sample(bat) |> device
        sc_frames = nothing
        if rand() < 0.5
            sc_frames, _ = model(ts.t, ts.Xt, ts.chainids, ts.resinds)
        end
        l, grad = Flux.withgradient(model) do m
            fr, aalogs = m(ts.t, ts.Xt, ts.chainids, ts.resinds, sc_frames = sc_frames)
            l_loc, l_rot, l_aas = losses(fr, aalogs, ts)
            l_loc + l_rot + l_aas
        end
        Flux.update!(opt_state, model, grad[1])
        if mod(i, 10) == 0
            Flux.adjust!(opt_state, next_rate(sched))
        end
        println(l)
    end
    jldsave("model_epoch_$epoch.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end