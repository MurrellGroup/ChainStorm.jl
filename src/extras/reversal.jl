using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../")

using Pkg
Pkg.add(["GLMakie", "ProtPlot", "ProgressBars", "CUDA", "cuDNN", "Flux"])

using ChainStorm, Flowfusion, GLMakie, ProtPlot, ProgressBars, CUDA, Flux

@eval Flowfusion begin
    function reverse_gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector, record; tracker::Function=Returns(nothing), midpoint = false, progress_bar=identity, snap_time = 0)
        Xₜ = copy.(X₀)
        push!(record, (1, X₀, nothing))
        for (s₁, s₂) in progress_bar(zip(steps, steps[begin+1:end]))
            T = eltype(s₁)
            s2 = s₂
            s1 = s₁ == s₂ ? s2 - T(0.001) : s₁
            t = midpoint ? (s1 + s2) / 2 : t = s1
            X0hat, X1hat = model(t, Xₜ)
            X0hat = resolveprediction(X0hat, Xₜ)
            X1hat = resolveprediction(X1hat, Xₜ)
            Xₜ = mask(step(P, Xₜ, X0hat, s1, s2), X₀)
            if t < snap_time
                fakeX1hat = deepcopy(X1hat)
                tensor(fakeX1hat[1]) .= tensor(X₀[1])
                tensor(fakeX1hat[2]) .= tensor(X₀[2])
                push!(record, (1-s₂, deepcopy(Xₜ), fakeX1hat)) #records all the steps
            else
                push!(record, (1-s₂, deepcopy(Xₜ), deepcopy(X1hat))) #records all the steps
            end
            tracker(1-t, Xₜ, X1hat)
        end
        return Xₜ
    end
    function gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector; tracker::Function=Returns(nothing), midpoint = false, progress_bar=identity)
        Xₜ = copy.(X₀)
        for (s₁, s₂) in progress_bar(zip(steps, steps[begin+1:end]))
            t = midpoint ? (s₁ + s₂) / 2 : t = s₁
            hat = resolveprediction(model(t, Xₜ), Xₜ)
            Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), X₀)
            tracker(t, Xₜ, hat)
        end
        return Xₜ
    end
    function bind_gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector, record; tracker::Function=Returns(nothing), midpoint = false, progress_bar = identity)
        Xₜ = copy.(X₀)
        for (s₁, s₂) in progress_bar(zip(steps, steps[begin+1:end]))
            t = midpoint ? (s₁ + s₂) / 2 : t = s₁
            hat = resolveprediction(model(t, Xₜ), Xₜ)
            #Changes xt 
            s₁, old_xt, _ = record[end]
            pop!(record)
            tensor(Xₜ[1])[:, :, 1:size(tensor(old_xt[1]), 3), :] .= tensor(old_xt[1])
            tensor(Xₜ[2])[:, :, 1:size(tensor(old_xt[2]), 3), :] .= tensor(old_xt[2])
            tensor(Xₜ[3]).indices[1:size(tensor(old_xt[3]).indices, 1), :] .= tensor(old_xt[3]).indices
            Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), X₀)
            if length(record) == 1
                tensor(Xₜ[1])[:, :, 1:size(tensor(old_xt[1]), 3), :] .= tensor(old_xt[1])
                tensor(Xₜ[2])[:, :, 1:size(tensor(old_xt[2]), 3), :] .= tensor(old_xt[2])
                tensor(Xₜ[3]).indices[1:size(tensor(old_xt[3]).indices, 1), :] .= tensor(old_xt[3]).indices
            end
            tracker(t, Xₜ, hat)
        end
        return Xₜ
    end
    export bind_gen, reverse_gen
end


model = load_model() |> gpu

struc = pdb"7RBY"1
target = ChainStorm.pdb2batch(struc[[1]])


@time rev_g, recorded = flow_quickgen(ChainStorm.reverse_process(ChainStorm.P), target, ChainStorm.compound_state(target), model, is_reverse = true, smooth=0, d = gpu, steps = 0f0:0.005f0:1f0, progress_bar = ProgressBar, snap_time = 0.9f0);

#proportions = [mean(unhot(recorded[i][2][3]).S.state .== 21) for i in 1:length(recorded)]
#pl = Plots.plot(proportions, xlabel = "t step", label = :none, ylabel = "P(21)")
#savefig(pl, "proportions.pdf")

b = dummy_batch([ChainStorm.lengths_from_chainids(target.chainids); [122, 114]])
b.resinds[1:length(target.resinds)] .= target.resinds
binder_inds = length(target.resinds)+1:length(b.resinds)
b.resinds[binder_inds] .= 1:length(binder_inds)
X0 = ChainStorm.zero_state(b)
#If you want to bias the starting location:
#Flowfusion.tensor(X0[1])[:,1,binder_inds,1] .*= 0.5f0
#Flowfusion.tensor(X0[1])[:,1,binder_inds,1] .+= [0.0f0, 0.1f0, 0.1f0]

paths = ChainStorm.Tracker()
@time fwd_g = flow_quickgen(ChainStorm.P, b, X0, model, tracker = paths, smooth=0, record = deepcopy(recorded), progress_bar = ProgressBar, d = gpu, steps = 0f0:0.005f0:1f0);
id = join(string.(ChainStorm.lengths_from_chainids(b.chainids)),"_")*"-"*join(rand('A':'Z', 4))
export_pdb("samples/$(id)_bind.pdb", fwd_g, b.chainids, b.resinds) #<- Save PDB

samp = gen2prot(fwd_g, b.chainids, b.resinds)
animate_trajectory("samples/$(id)_bind.mp4", samp, first_trajectory(paths), viewmode = :fit)

for _ in 1:10
    snap = rand([0f0, 0.9f0])
    @time rev_g, recorded = flow_quickgen(ChainStorm.reverse_process(ChainStorm.P), target, ChainStorm.compound_state(target), model, is_reverse = true, smooth=0, d = gpu, steps = 0f0:0.005f0:1f0, progress_bar = ProgressBar, snap_time = snap); #, steps = 0f0:0.025f0:1f0); #<- Model inference call
    lens = [200+rand(1:25),200+rand(1:25)]
    if rand() < 0.33
        lens = [100+rand(1:25),100+rand(1:25)]
    end
    if rand() < 0.33
        lens = [rand(30:150)]
    end
    b = dummy_batch([ChainStorm.lengths_from_chainids(target.chainids); lens])
    b.resinds[1:length(target.resinds)] .= target.resinds
    binder_inds = length(target.resinds)+1:length(b.resinds)
    b.resinds[binder_inds] .= 1:length(binder_inds)
    X0 = ChainStorm.zero_state(b)
    paths = ChainStorm.Tracker()
    @time fwd_g = flow_quickgen(ChainStorm.P, b, X0, model, tracker = paths, smooth=0, record = deepcopy(recorded), progress_bar = ProgressBar, d = gpu, steps = 0f0:0.005f0:1f0);
    id = join(string.(ChainStorm.lengths_from_chainids(b.chainids)),"_")*"-"*join(rand('A':'Z', 4))
    export_pdb("samples/$(id)_$(snap)_bind.pdb", fwd_g, b.chainids, b.resinds) #<- Save PDB    
    samp = gen2prot(fwd_g, b.chainids, b.resinds)
    animate_trajectory("samples/$(id)_$(snap)_bind.mp4", samp, first_trajectory(paths), viewmode = :fit)
end