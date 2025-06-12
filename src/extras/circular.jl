
struct WrappedInd{A,B}
    ind::A
    max::B
end
function Base.:-(a::WrappedInd, b::WrappedInd)
    direct = a.ind - b.ind
    wrapped = sign(direct) * (a.max - abs(direct))
    return abs(direct) < abs(wrapped) ? direct : -wrapped
end

"""
    circularize(batch, circular_chain_ids::AbstractVector)

Takes a batch (eg. returned by dummy_batch) and a list of numeric chain ids (or a single chain id) that is intended to be circularized.
Returns the same batch, but with residue indices for the selected chains replaced by WrappedInd types, which will circularize.

```julia
b = dummy_batch([15,60])
wrapped_b = circularize(b, 1)
g = flow_quickgen(wrapped_b, model)
export_pdb("test.pdb", g, b.chainids, b.resinds) #<- Note: original resinds are used
```
"""
function circularize(batch, circular_chain_ids::AbstractVector)
    max_lengths = 1000000 .* ones(Int, size(batch.chainids))
    for c in circular_chain_ids
        inds = batch.chainids .== c
        max_lengths[inds] .= maximum(batch.resinds[inds])
    end
    return merge(batch, (; resinds = WrappedInd.(batch.resinds, max_lengths)))
end
circularize(batch, circular_chain_ids::Integer) = circularize(batch, [circular_chain_ids])
