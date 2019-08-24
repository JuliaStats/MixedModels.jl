using DataFrames, FreqTables, MixedModels, Random, StatsModels
#using Test

const LMM = LinearMixedModel;
const GLMM = GeneralizedLinearMixedModel;
const HC = HelmertCoding();
const contrasts = Dict(:A=>HC,:B=>HC,:C=>HC);

const rng = Random.MersenneTwister(4321234);

const LETTERS = ['A':'Z';];
const letters = ['a':'z';];
const Letters = vcat(LETTERS, letters);
const plusminus = ['+','-'];
const dat = (
    A = repeat(plusminus, outer=4096),
    B = repeat(plusminus, inner=2, outer=2048),
    C = repeat(plusminus, inner=4, outer=1024),
    G = repeat(Letters[1:32], inner=8, outer=32),
    H = repeat(letters[1:16], inner=16, outer=32),
    I = repeat(Letters[1:32], inner=256),
    Y = ones(8192),
);
show(DataFrame(dat))
println()
println(freqtable(dat, :H, :G))  # G is nested within H
println(freqtable(dat, :I, :G))  # G and I are crossed
println(freqtable(dat, :A, :G))
println(freqtable(dat, :B, :G))
println(freqtable(dat, :C, :G))
println(freqtable(dat, :A, :I))
println(freqtable(dat, :B, :I))
println(freqtable(dat, :C, :I))
m1 = LMM(@formula(Y ~ 1 + A*B*C + (1|G) + (1|I)), dat, contrasts);
β = rand(rng, 8);
show(refit!(simulate!(rng, m1, β=β, σ=0.4, θ=[0.8, 0.6])))
println()
@show(β);
println()
show(m1.optsum);

m2 = LMM(@formula(Y ~ 1 + A*B*C + (1+A+B+C|G) + (1|I)), dat, contrasts);
show(refit!(m2, copy(response(m1))))
m3 = LMM(@formula(Y ~ 1 + A*B*C + (1+A+B+C|G) + (1+A+B+C|I)), dat, contrasts);
show(refit!(m3, copy(response(m1))))
m4 = LMM(@formula(Y ~ 1 + A*B*C + (1+A*B*C|G) + (1+A*B*C|I)), dat, contrasts);
show(refit!(m4, copy(response(m1))))

logistic(η) = inv(1 + exp(-η))
"""
    bresp(rng::AbstractRNG, v::Vector{<:AbstractFloat})

Return a random sample from `[Bernoulli(logistic(x)) for x in v]`
"""
bresp(rng::AbstractRNG, v::AbstractVector{<:AbstractFloat}) =
    rand(rng, length(v)) .< logistic.(v)

copyto!(dat.Y, bresp(rng, response(m1)))
g1 = GLMM(@formula(Y ~ 1 + A*B*C + (1|G) + (1|H)), dat, Bernoulli(), contrasts=contrasts)
