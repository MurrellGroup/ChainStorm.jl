using ChainStorm
using Test

@testset "ChainStorm.jl" begin
    chainlengths = [5,5]
    b = dummy_batch(chainlengths)
    X0 = ChainStorm.zero_state(b)
    X1 = ChainStorm.zero_state(b)
    t = rand(Float32, 1, size(b.aas,2))
    Xt = ChainStorm.bridge(P, X0, X1, t)
    rotξ = ChainStorm.Guide(Xt[2], X1[2])
    ts = (; t, Xt, X1, rotξ, chainids = b.chainids, resinds = b.resinds)
    m = ChainStormV1(16, 2, 2)
    fr, aalogs = m(ts.t, ts.Xt, ts.chainids, ts.resinds)
    l_loc, l_rot, l_aas = losses(fr, aalogs, ts)
    @test isa(l_loc, Float32)
    @test isa(l_rot, Float32)
    @test isa(l_aas, Float32)
end
