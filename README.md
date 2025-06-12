# ChainStorm

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ChainStorm.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ChainStorm.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ChainStorm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ChainStorm.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ChainStorm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ChainStorm.jl)


This repo implements a structure/sequence co-design model, using diffusion/flow matching (from [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl)) with an architecture based primarily on AlphaFold 2's Invariant Point Attention (here via [InvariantPointAttention.jl](https://github.com/MurrellGroup/InvariantPointAttention.jl)). The protein backbone is represented as a sequence of "frames", each with a location and rotation, as well as a discrete amino acid character. The model is trained to take noised input (where the locations, rotations, and discrete states have all been perturbed, to a random degree, by a noising process) and predict the original (ie. un-noised) protein structure. With a model thus trained, samples from the distribution of training structures can be sampled by taking many small steps from a random starting distribution.

<video src="https://github.com/user-attachments/assets/4cef2445-d4e6-4d6c-9e50-1b99f79bb9a4" controls></video>

## Installation

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
#Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
Pkg.add(url = "https://github.com/MurrellGroup/ChainStorm.jl")
```

## Quick start

This will load up a model and generate a single small protein with two chains, each of length 20:

```julia
using ChainStorm
model = load_model()
b = dummy_batch([20,20]) #<- The model's only input
g = flow_quickgen(b, model) #<- Model inference call
export_pdb("gen.pdb", g, b.chainids, b.resinds) #<- Save PDB
```

## Visualization, and using the GPU

```julia
using Pkg
Pkg.add(["GLMakie", "ProtPlot"])

using ChainStorm, GLMakie, ProtPlot

#If GPU:
using CUDA
dev = ChainStorm.gpu
#device = identity #<- If no GPU

model = load_model() |> dev

chainlengths = [54,54]
b = dummy_batch(chainlengths)
paths = ChainStorm.Tracker() #The trajectories will end up in here
g = flow_quickgen(b, model, d = dev, tracker = paths) #<- Model inference call
id = join(string.(chainlengths),"_")*"-"*join(rand('A':'Z', 4))
export_pdb("$(id).pdb", g, b.chainids, b.resinds) #<- Save PDB
samp = gen2prot(g, b.chainids, b.resinds)
animate_trajectory("$(id).mp4", samp, first_trajectory(paths), viewmode = :fit) #<- Animate design process
```

Note: If you need the animations via GLMakie to run headless, in linux you can install xvfb, then run these in the terminal before starting your Julia session/script:
```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

## Training

```julia
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
```