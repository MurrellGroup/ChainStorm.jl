
using Pkg
Pkg.activate(".")

rundir = "ChainStorm_AASC1"
mkpath(rundir)
GPUnum = 1
ENV["CUDA_VISIBLE_DEVICES"] = GPUnum

Pkg.add("Revise")
using Revise

Pkg.develop(path="../")

#Pkg.add(["DiscreteProteinSandwich", "ProteinChains", "MergedArrays", "InvariantPointAttention", "Flux"])
#Pkg.add(url = "https://github.com/MurrellGroup/Onion.jl")
#Pkg.add(url = "https://github.com/MurrellGroup/DLProteinFormats.jl")

using ChainStorm, DLProteinFormats, ProteinChains, InvariantPointAttention, Flux, Onion, CUDA, CannotWaitForTheseOptimisers, LearningSchedules, JLD2
using DLProteinFormats: PDBSimpleFlat500, sample_batched_inds, length2batch, batch_flatrecs

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu 

dat = DLProteinFormats.load(PDBSimpleFlat);

#Folding in the self-cond layers:
#=
model_state = JLD2.load("proflowdemo_run1after5epochnoselfcond_checkpoint_20.jld2", "model_state")
oldmodel = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(oldmodel, model_state)
newishmodel = FlowcoderAASC(384, 6, 6)
model = FlowcoderAASC(merge(newishmodel.layers, oldmodel.layers)) |> device
#model = ChainStorm(merge(newishmodel.layers, oldmodel.layers)) |> device
#make the selfcond IPA output have a small effect, initially:
for transformer in model.layers.transformers
    transformer.ipa.layers.ipa_linear.weight ./= 10
end
=#

opt_state = Flux.setup(Muon(eta = old_lr_schedule_state.lr), model); #Did I re-init the opt state for the warmdown??
sched = burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.999975f0)
#sched = linear_decay_schedule(sched.lr, 0.000000001f0, 3414) #Two epoch warmdown

ls = Float32[]
for epoch in 1:100
    if epoch == 5
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 3414) #Two epoch warmdown
    end
    batchinds = DLProteinFormats.sample_batched_inds(dat, l2b = DLProteinFormats.length2batch(1500, 1.9))
    for s in 1:20
        b = batch_flatrecs(dat[[rand(vcat(batchinds...))]])
        if rand() < 0.75
            b.resinds[:] .= 1:length(b.resinds)
        end
        stps = rand(100:1000)
        export_pdb("$(rundir)/$(epoch)_$(s)_sample_$(stps).pdb", flow_quickgen(b, model, d = gpu, steps = stps), b.chainids, b.resinds)
    end
    tot_l = 0f0
    for i in 1:length(batchinds)
        b = batch_flatrecs(dat[batchinds[i]])
        ts = training_sample(b) |> device
        X1hat_frames = nothing
        if rand() < 0.5
            X1hat_frames, X1hat_aalogits = model(ts.t, ts.Xt, ts.chainids, ts.resinds)
        end
        l,grad = Flux.withgradient(model) do m
            hatframes, aalogits = m(ts.t, ts.Xt, ts.chainids, ts.resinds, sc_frames = X1hat_frames, sc_aa = X1hat_aalogits)
            l_loc, l_rot, l_aa = losses(hatframes, aalogits, ts)
            Flux.Zygote.@ignore mod(i, 200) == 0 && println("loc: ", l_loc, ", rot: ", l_rot, ", aa: ", l_aa, " t vec: ", ts.t)
            l_loc + l_rot + l_aa
        end
        Flux.update!(opt_state, model, grad[1])
        tot_l = tot_l+l
        if mod(i, 10) == 0
            Flux.adjust!(opt_state, next_rate(sched))
            println("L: ", tot_l/10,"; epoch: ", epoch, ", i: ", i, " of ", length(batchinds), "; lr: ", sched.lr)
            tot_l = 0f0
        end
        push!(ls, l)
    end
    mod(epoch,1) == 0 && jldsave("$(rundir)/checkpoint_$(epoch).jld2"; model_state=Flux.state(cpu(model)), opt_state=cpu(opt_state), epoch=epoch, lr_schedule = sched)
end
