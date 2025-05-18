# ChainStorm

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ChainStorm.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ChainStorm.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ChainStorm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ChainStorm.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ChainStorm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ChainStorm.jl)


This repo implements a structure/sequence co-design model, using diffusion/flow matching (from [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl)) with an architecture based primarily on AlphaFold 2's Invariant Point Attention (here via [InvariantPointAttention.jl](https://github.com/MurrellGroup/InvariantPointAttention.jl)). The protein backbone is represented as a sequence of "frames", each with a location and rotation, as well as a discrete amino acid character. The model is trained to take noised input (where the locations, rotations, and discrete states have all been perturbed, to a random degree, by a noising process) and predict the original (ie. un-noised) protein structure. With a model thus trained, samples from the distribution of training structures can be sampled by taking many small steps from a random starting distribution.

<video src="https://github.com/user-attachments/assets/4cef2445-d4e6-4d6c-9e50-1b99f79bb9a4" controls></video>

## Quick start

This will load up a model and generate a single small protein with two chains, each of length 20:

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.activate(".")
Pkg.add(["JLD2", "Flux", "HuggingFaceApi"])
Pkg.add(url = "https://github.com/MurrellGroup/ChainStorm.jl")

using ChainStorm, JLD2, Flux, HuggingFaceApi

file = hf_hub_download("MurrellLab/ChainStorm", "ChainStorm_chkpt_3.jld2");
model = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(model, JLD2.load(file, "model_state"));

b = dummy_batch([20,20]) #<- The lengths of each chain are the model's only input
g = flow_quickgen(b, model) #<- Model inference call
export_pdb("gen.pdb", g, b.chainids, b.resinds) #<- Save PDB
```

## Installation

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.add(["HuggingFaceApi", "JLD2", "Flux"]) #<- For fetching and loading weights
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
Pkg.add(url = "https://github.com/MurrellGroup/ChainStorm.jl")
```

## Visualization, and using the GPU

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.activate(".")
Pkg.add(["JLD2", "Flux", "GLMakie", "ProtPlot", "HuggingFaceApi"])
Pkg.add(url = "https://github.com/MurrellGroup/ChainStorm.jl")

using ChainStorm, JLD2, Flux, HuggingFaceApi
using GLMakie, ProtPlot

#If GPU:
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
using CUDA
device = gpu

#device = identity #<- If no GPU

file = hf_hub_download("MurrellLab/ChainStorm", "ChainStorm_chkpt_3.jld2");
model = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(model, JLD2.load(file, "model_state"));
model = model |> device

chainlengths = [124,124] #<- The model's only input
b = dummy_batch(chainlengths)
paths = ChainStorm.Tracker()
g = flow_quickgen(b, model, d = device, tracker = paths) #<- Model inference call
id = join(string.(chainlengths),"_")*"-"*join(rand("0123456789ABCDEFG", 4))
export_pdb("$(id).pdb", g, b.chainids, b.resinds) #<- Save PDB
samp = gen2prot(g, b.chainids, b.resinds)
animate_trajectory("$(id).mp4", samp, first_trajectory(paths), viewmode = :fit) #<- Animate design process
```
