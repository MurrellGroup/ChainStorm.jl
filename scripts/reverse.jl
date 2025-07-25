using Pkg
Pkg.add(["GLMakie", "ProtPlot", "ProgressBars"])

using ChainStorm, Flowfusion, GLMakie, ProtPlot, ProgressBars

@eval Flowfusion begin
    function reverse_gen(P::Tuple{Vararg{UProcess}}, X₀::Tuple{Vararg{UState}}, model, steps::AbstractVector, record; tracker::Function=Returns(nothing), midpoint = false, progress_bar=identity)
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

            push!(record, (1-s₂, Xₜ, X1hat)) #records all the steps
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


model = load_model();

b = ChainStorm.pdb2batch("path/to/pdb")

g, recorded = flow_quickgen(ChainStorm.reverse_process(ChainStorm.P), b, ChainStorm.compound_state(b), model, is_reverse = true, smooth=0, progress_bar = ProgressBar) #<- Model inference call

b = dummy_batch([ChainStorm.lengths_from_chainids(b.chainids); [10]])
paths = ChainStorm.Tracker()
g = flow_quickgen(ChainStorm.P, b, ChainStorm.zero_state(b), model, tracker = paths, smooth=0, record = recorded, progress_bar = ProgressBar)
id = join(string.(ChainStorm.lengths_from_chainids(b.chainids)),"_")*"-"*join(rand('A':'Z', 4))
export_pdb("$(id)_bind.pdb", g, b.chainids, b.resinds) #<- Save PDB
samp = gen2prot(g, b.chainids, b.resinds)
animate_trajectory("$(id)_bind.mp4", samp, first_trajectory(paths), viewmode = :fit)