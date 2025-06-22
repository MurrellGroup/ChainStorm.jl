using Pkg
Pkg.activate(".")

rundir = "ChainStormDesign_1"
mkpath(rundir)
mkpath("$(rundir)/samples")
GPUnum = 0
ENV["CUDA_VISIBLE_DEVICES"] = GPUnum

Pkg.add("Revise")
using Revise

Pkg.develop(path="../")

#Pkg.add(["DLProteinFormats", "InvariantPointAttention", "Flux", "CannotWaitForTheseOptimisers", "LearningSchedules", "JLD2"])
#Pkg.add(url = "https://github.com/MurrellGroup/Onion.jl")

using ChainStorm, DLProteinFormats, ProteinChains, Flux, CUDA, CannotWaitForTheseOptimisers, LearningSchedules, JLD2, HuggingFaceApi, Plots
using DLProteinFormats: PDBSimpleFlat500, sample_batched_inds, length2batch, batch_flatrecs

device!(0) #Because we have set CUDA_VISIBLE_DEVICES = GPUnum
device = gpu 


function att_plot(att, chainids, path, cm = cgrad(:linear_kryw_5_100_c67_n256, rev = true))
    pl2 = heatmap(att, size = (250,250), c = cm, colorbar = :none, clim = (0,Inf), xlim = (0,size(att, 1)+1), ylim = (0,size(att, 2)+1), tick_direction = :out)
    chainlengths = [sum(chainids .== i) for i in unique(chainids)]
    cusu = vcat([0],cumsum(chainlengths))
    L = size(att, 1)
    for i in 1:length(cusu)
        plot!([0.5, L+0.5], [cusu[i]+0.5, cusu[i]+0.5], label = :none, linestyle = :dash, color = :blue, alpha = 0.5)
        plot!([cusu[i]+0.5, cusu[i]+0.5], [0.5, L+0.5], label = :none, linestyle = :dash, color = :blue, alpha = 0.5)
    end
    savefig(pl2, path)
end


file = hf_hub_download("MurrellLab/ProFlowDemo", "ProFlowDemo_chkpt_31.jld2");
oldmodel = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(oldmodel, JLD2.load(file, "model_state"));
newmodel = ChainStormDesign(384, 6, 6)
model = ChainStormDesign(merge(newmodel.layers, oldmodel.layers)) |> device

sched = burnin_learning_schedule(0.00001f0, 0.001f0, 1.02f0, 0.999975f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model); 

Flux.freeze!(opt_state)
Flux.thaw!(opt_state.layers.design_project)
Flux.thaw!(opt_state.layers.pair_project)
for layer in opt_state.layers.selfcond_crossipa
    Flux.thaw!(layer.ipa.layers.pair)
end
for layer in opt_state.layers.selfcond_selfipa
    Flux.thaw!(layer.ipa.layers.pair)
end
for block in opt_state.layers.ipa_blocks
    Flux.thaw!(block.ipa.layers.pair)
end

dat = DLProteinFormats.load(PDBSimpleFlat);

ls = Float32[]
for epoch in 1:100
    if epoch == 2
        Flux.thaw!(opt_state)
        sched = burnin_learning_schedule(0.000001f0, 0.0005f0, 1.05f0, 0.999975f0)
        Flux.adjust!(opt_state, sched.lr)
    end
    if epoch == 6
        sched = linear_decay_schedule(sched.lr, 0.000000001f0, 6500) #Four epoch warmdown
    end
    batchinds = DLProteinFormats.sample_batched_inds(dat, l2b = DLProteinFormats.length2batch(1600, 1.9))
    numsamples = epoch == 1 ? 2 : 10
    for s in 1:numsamples        
        prefix = "$(rundir)/samples/$(epoch)_$(s)_cond"
        b = batch_flatrecs(dat[[rand(vcat(batchinds...))]])
        temp_ts = training_sample(b)
        cond_mask, cond_noise_mult = rand_conditioning_mask(temp_ts.chainids) |> device
        pair_conditioning = random_distance_features(ChainStorm.tensor(device(temp_ts.X1[1])),cond_mask,cond_noise_mult)
        export_pdb("$(prefix).pdb", flow_quickgen(b, model, d = gpu, pair_cond = pair_conditioning), b.chainids, b.resinds)
        att_plot(cpu(pair_conditioning[1,:,:,1]), temp_ts.chainids[:,1], "$(prefix)_dists.png")
        att_plot(cpu(pair_conditioning[2,:,:,1]), temp_ts.chainids[:,1], "$(prefix)_noise.png")
        att_plot(cpu(pair_conditioning[3,:,:,1]), temp_ts.chainids[:,1], "$(prefix)_mask.png")
    end
    for s in (numsamples+1):numsamples*2        
        l1, l2 = rand(50:300), rand(50:300)
        chainlengths = [l1, l1, l2, l2]
        prefix = "$(rundir)/samples/$(epoch)_$(s)_nocond_$(join(string.(chainlengths), "-"))"
        b = dummy_batch(chainlengths)
        g = flow_quickgen(b, model, d = device)
        export_pdb("$(prefix).pdb", g, b.chainids, b.resinds)
    end

    tot_l = 0f0
    for i in 1:length(batchinds)
        b = batch_flatrecs(dat[batchinds[i]])
        cpu_ts = training_sample(b)
        pair_conditioning = nothing
        ts = cpu_ts |> device
        if rand() < 0.5
            cond_mask, cond_noise_mult = rand_conditioning_mask(cpu_ts.chainids) |> device
            pair_conditioning = random_distance_features(ChainStorm.tensor(ts.X1[1]),cond_mask,cond_noise_mult)
        end
        X1hat_frames = nothing
        if rand() < 0.5
            X1hat_frames, X1hat_aalogits = model(ts.t, ts.Xt, ts.chainids, ts.resinds; pair_conditioning)
        end
        l,grad = Flux.withgradient(model) do m
            hatframes, aalogits = m(ts.t, ts.Xt, ts.chainids, ts.resinds; sc_frames = X1hat_frames, pair_conditioning)
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