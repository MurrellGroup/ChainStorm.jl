using Dates
using ChainStorm, DLProteinFormats, Flux, CannotWaitForTheseOptimisers, LearningSchedules, JLD2
using DLProteinFormats: load, PDBSimpleFlat, batch_flatrecs, sample_batched_inds, length2batch
using CUDA, cuDNN

fallback_mask(x) = any(size(x) .== 21)

device = gpu
dat = load(PDBSimpleFlat);
sample = dat[9]
print(sample.len)
L = length(sample.chainids)
c = (;chainids = reshape(sample.chainids, :, 1), 
    resinds = view(sample.resinds, :, 1), 
    padmask = trues(L, 1), 
    aas = reshape(sample.AAs, :, 1), 
    locs = reshape(sample.locs, 3, 1, L, 1))

model = ChainStorm.load_model() |> device
model_cpu = nothing

sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.999995f0);
opt_state = Flux.setup(Muon(eta = sched.lr, fallback = fallback_mask), model);

start_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")

gen_sample_counter = 0
sample_counter = 10000
for epoch in 1:100
    batchinds = sample_batched_inds(dat,l2b = length2batch(1500, 1.9))
    @info "Epoch $epoch"
    for (i, b) in enumerate(batchinds)
        sample_counter += 1
        bat = batch_flatrecs(dat[b])
        ts = training_sample(bat) |> device
        sc_frames = nothing
        if epoch > 1 && rand() < 0.5
            sc_frames = model(ts.t, ts.Xt, ts.aas, ts.chainids, ts.resinds)
        end
        l, grad = Flux.withgradient(model) do m
            fr = m(ts.t, ts.Xt, ts.aas, ts.chainids, ts.resinds, sc_frames = sc_frames)
            l_loc, l_rot = losses(fr, ts)
            l_loc + l_rot 
        end
        Flux.update!(opt_state, model, grad[1])
        (mod(i, 10) == 0) && Flux.adjust!(opt_state, next_rate(sched))
        println(l)
        println("Sample counter: ", sample_counter, " Gen sample counter: ", gen_sample_counter)    
        if sample_counter >= 10000
            sample_counter = 0
            gen_sample_counter += 1
            model_cpu = cpu(model)
            g = flow_quickgen(c, model_cpu)
            export_pdb("notrunk_toymodel/gens/gen_$(start_time)_foldfinetune_sample_$(gen_sample_counter).pdb", (g..., ChainStorm.seq_to_masked_state(sample.AAs)), c.chainids, c.resinds)
        end
    end
    jldsave("model_epoch_$epoch.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end
