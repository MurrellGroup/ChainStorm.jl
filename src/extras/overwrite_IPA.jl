@eval ChainStorm.InvariantPointAttention begin

#=
Attention maps get pushed to `task_local_storage(:attention_container)::Vector{Any}` if
`task_local_storage(:save_pass, true)` is set. Deactivate by setting `false` instead.

The attention maps can be transformed by setting `task_local_storage(:attention_transform, f)`,
e.g. for selecting a single head or returning nothing.
=#

function (ipa::Union{IPCrossA, IPA})(
    TiL::Tuple{AbstractArray, AbstractArray}, siL::AbstractArray,
    TiR::Tuple{AbstractArray, AbstractArray}, siR::AbstractArray;
    zij = nothing, mask = 0, customgrad = false, 
    rope::Union{IPARoPE, Nothing} = nothing, chain_diffs = 1, show_warnings = false
)
    if !isnothing(zij)
        #This is assuming the dims of zij are c, N_frames_L, N_frames_R, batch
        size(zij,2) == size(siR,2) || throw(DimensionMismatch("zij and siR size mismatch"))
        size(zij,3) == size(siL,2) || throw(DimensionMismatch("zij and siL size mismatch")) 
    end
    if mask != 0
        size(mask,1) == size(siR, 2) || throw(DimensionMismatch("mask and siR size mismatch"))
        size(mask,2) == size(siL, 2) || throw(DimensionMismatch("mask and siL size mismatch"))
    end
    l = ipa.layers
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings
    if haskey(ipa.settings, :use_softmax1) #For compat
        use_softmax1 = ipa.settings.use_softmax1
    else
        use_softmax1 = false
    end
    rot_TiL, translate_TiL = TiL
    rot_TiR, translate_TiR = TiR
    N_frames_L = size(siL,2)
    N_frames_R = size(siR,2)
    gamma_h = softplus(clamp.(l.gamma_h,Typ(-100), Typ(100))) #Clamping
    w_C = Typ(sqrt(2/(9*N_query_points)))
    dim_scale = Typ(1/sqrt(c))    
    qh = reshape(l.proj_qh(siR),(c,N_head,N_frames_R,:))
    kh = reshape(l.proj_kh(siL),(c,N_head,N_frames_L,:))
    if !isnothing(rope)
        qhTkh = dotproducts(rope, qh, kh; chain_diffs)
    else
        qhTkh = dotproducts(qh, kh)
    end
    vh = reshape(l.proj_vh(siL),(c,N_head,N_frames_L,:))
    if isnothing(l.scale_h)
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    else
        scale_h = reshape(l.scale_h, (1,N_head*N_query_points,1,1))
        qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:)) .* scale_h
        khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:)) .* scale_h
    end
    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))
    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)
    diffs_glob = Flux.unsqueeze(Tqhp, dims = 5) .- Flux.unsqueeze(Tkhp, dims = 4)
    sum_norms_glob = reshape(sum(abs2, diffs_glob, dims = [1,3]),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    att_arg = reshape(dim_scale .* qhTkh .- w_C/2 .* gamma_h .* sum_norms_glob,(N_head,N_frames_R,N_frames_L, :))
    flatt_arg = reshape(dim_scale .* qhTkh,(N_head,N_frames_R,N_frames_L, :))
    if pairwise
        w_L = Typ(sqrt(1/3))
        bij = reshape(l.pair(zij),(N_head,N_frames_R,N_frames_L,:))
    else
        w_L = Typ(sqrt(1/2))
        bij = Typ(0)
    end
    # Setting mask to the correct dim for broadcasting. 
    if mask != 0 
        mask = Flux.unsqueeze(mask, dims = 1) 
    end

    softmax = use_softmax1 ? softmax1 : Flux.softmax
    att = softmax(w_L .* (att_arg .+ bij) .+ mask, dims=3)
    att_no_frame = softmax(w_L .* (flatt_arg .+ bij) .+ mask, dims=3)

    ## <ATTENTION>: THIS IS, TRACKING CODE (with yoda voice)
    if get(task_local_storage(), :save_pass, true)
        container = get!(task_local_storage(), :attention_container, [])
        transform = get(task_local_storage(), :attention_transform, Returns(nothing))
        x = (att=transform(Float16.(att)), att_no_frame=transform(Float16.(att_no_frame))) # H x L x L x B
        push!(container, x)
    end
    ## </ATTENTION>

    # Applying the attention weights to the values.
    oh = permutedims(batched_mul(permutedims(att,(2,3,1,4)), permutedims(vh,(3,1,2,4))),(2,3,1,4));
    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))
    if use_softmax1
        pre_ohp_r = sum(broadcast_att_ohp.*broadcast_tvhp,dims=5)
        unreshaped_ohp_r = pre_ohp_r .+ (1 .- sum(broadcast_att_ohp, dims = 5)) .* reshape(translate_TiR, 3, 1, 1, N_frames_R, 1, :)
        ohp_r = reshape(unreshaped_ohp_r,3,N_head*N_point_values,N_frames_R,:)
    else
        ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)
    end
    #ohp_r were in the global frame, so we put those ba ck in the recipient local
    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) 
    normed_ohp = sqrt.(sum(abs2, ohp,dims = 1) .+ Typ(0.000001f0)) #Adding eps
    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        ) 
    if pairwise
        obh = batched_mul(permutedims(zij,(1,3,2,4)), permutedims(att,(3,1,2,4)))
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames_R,:))
    end
    si = l.ipa_linear(catty) 
    return si 
end

end
