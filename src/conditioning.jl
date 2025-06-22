function batched_pairwise_dist(x)
    sqnorms = sum(abs2, x, dims=1)
    A_sqnorms = reshape(sqnorms, size(x, 2), 1, size(x, 3))
    B_sqnorms = reshape(sqnorms, 1, size(x, 2), size(x, 3))
    return A_sqnorms .- 2 * (batched_transpose(x) ⊠ x) .+ B_sqnorms
end

batched_pairwise_dist(x::AbstractArray{T,4}) where T = batched_pairwise_dist(reshape(x, 3, size(x, 3), size(x, 4)))

#Since zero is "no info", we scale the distances between 0.5 (really close) to 1 (far away):
dist_transform(x) = 1 ./ (1 .+ exp.(-x))

function rand_conditioning_mask(initial_chainids)
    noise_scale = 0.5f0
    chainids = copy(initial_chainids)
    bat = size(chainids, 2)
    l = size(chainids, 1)
    mask = zeros(Bool, l, l, bat) #When the mask is 1, the pairwise distance is included. Else it will be set to 0 and ignored.
    noise_mult = zeros(Float32, l, l, bat)
    for b in 1:bat
        chains = union(chainids[:,b])
        max_chainid = maximum(chainids[:,b])
        for c in chains
            if rand() < 0.2 #Split the chain into two different chains, for the purposes of design masks etc
                c_inds = findall(chainids[:,b] .== c)
                splitpos = rand(1:length(c_inds))
                chainids[c_inds[splitpos:end],b] .= max_chainid + 1
                max_chainid += 1
            elseif rand() < 0.2 #Split chain into three chains
                c_inds = findall(chainids[:,b] .== c)
                splitpos1, splitpos2 = sort([rand(1:length(c_inds)), rand(1:length(c_inds))])
                chainids[c_inds[splitpos1:end],b] .= max_chainid + 1
                chainids[c_inds[splitpos2:end],b] .= max_chainid + 2
                max_chainid += 2
            end
        end
        chains = union(chainids[:,b])
        conditioned_chains = chains[rand(length(chains)) .< 0.5]
        for c in conditioned_chains
            c_inds = findall(chainids[:,b] .== c)
            mask[c_inds, c_inds, b] .= true
            chain_noise_mult = rand() < 0.5 ? noise_scale * rand() : 0
            noise_mult[c_inds, c_inds, b] .= chain_noise_mult
        end
        #then add some pairs to the mask:
        for c1 in conditioned_chains
            for c2 in conditioned_chains
                if c1 != c2
                    if rand() < 0.5
                        c1_inds = findall(chainids[:,b] .== c1)
                        c2_inds = findall(chainids[:,b] .== c2)
                        mask[c1_inds, c2_inds, b] .= true
                        mask[c2_inds, c1_inds, b] .= true
                        chain_pair_noise_mult = rand() < 0.5 ? noise_scale *rand() : 0
                        noise_mult[c1_inds, c2_inds, b] .= chain_pair_noise_mult
                        noise_mult[c2_inds, c1_inds, b] .= chain_pair_noise_mult
                    end
                end
            end
        end
        #then add some random runs for some random chains, where we flip the mask
        for c in chains
            if rand() < 0.25
                c_inds = findall(chainids[:,b] .== c)
                run_mask = zeros(Bool, length(c_inds))
                flip_start, flip_end = sort([rand(1:length(c_inds)),rand(1:length(c_inds))])
                run_mask[flip_start:flip_end] .= true
                if rand() < 0.5
                    run_mask .= .!run_mask
                end
                mask[c_inds, c_inds, b] .= run_mask .& run_mask'
                chain_noise_mult = rand() < 0.5 ? noise_scale * rand() : 0
                noise_mult[c_inds, c_inds, b] .= chain_noise_mult
            end
        end
        #then add some random pairwise masks
        if rand() < 0.25
            for i in 1:rand(1:10)
                p1,p2 = rand(1:l), rand(1:l)
                mask[p1,p2,b] = true
                mask[p2,p1,b] = true
                chain_pair_noise_mult = rand() < 0.5 ? noise_scale *rand() : 0
                noise_mult[p1,p2,b] = chain_pair_noise_mult
                noise_mult[p2,p1,b] = chain_pair_noise_mult
            end
        end
    end
    return mask .+ 0f0, noise_mult
end


#Note: this should be consistent during self-conditioning
function random_distance_features(locs, m, n) #Coords, mask, noise_multiplier
    pd = batched_pairwise_dist(locs)
    dists = dist_transform(pd)
    noised_dists = (dists .+ randn!(similar(dists)) .* n) .* m
    return vcat(reshape(noised_dists, 1, size(noised_dists)...), reshape(n, 1, size(n)...), reshape(m, 1, size(m)...))
end

#@time m, n = ChainStorm.rand_conditioning_mask(chainids);