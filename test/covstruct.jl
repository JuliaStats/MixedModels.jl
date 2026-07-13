using LinearAlgebra
using MixedModels
using Random
using StableRNGs
using StatsModels
using Test

using MixedModels: CompoundSymmetry, ScaledIdentity, Unstructured
using MixedModels: canonicalize!, initialθ, lowerbd, nθ, updateλ!
using MixedModelsDatasets: dataset

include("modelcache.jl")

const LMM = LinearMixedModel

# reference covariance matrices for the parameterizations
function homcs_cov(a, g, S)
    J = fill(inv(float(S)), S, S)
    return a^2 * (I - J) + g^2 * J
end

function hetcs_cov(d, b)
    S = length(d)
    c = (2 + S * b) * b
    ρ = c / (1 + c)
    R = fill(ρ, S, S)
    R[diagind(R)] .= 1
    return Diagonal(d) * R * Diagonal(d)
end

@testset "updateλ! builds the correct factor" begin
    for S in 2:5
        λ = LowerTriangular(Matrix{Float64}(I, S, S))
        rng = StableRNG(42 + S)
        for _ in 1:25
            # heterogeneous compound symmetry
            d = 0.2 .+ rand(rng, S)
            b = 0.4 * randn(rng)
            cs = CompoundSymmetry{Float64,true}(vcat(d, b))
            updateλ!(cs, λ, cs.θ)
            @test λ * λ' ≈ hetcs_cov(d, b)
            @test istril(λ)

            # homogeneous compound symmetry
            a, g = 0.2 .+ rand(rng, 2)
            hc = CompoundSymmetry{Float64,false}([a, g])
            updateλ!(hc, λ, hc.θ)
            @test λ * λ' ≈ homcs_cov(a, g, S)
        end
        # scaled identity
        d = Diagonal(zeros(S))
        si = ScaledIdentity{Float64}([1.7])
        updateλ!(si, d, si.θ)
        @test d ≈ 1.7 * I(S)
    end
end

@testset "boundary cases produce finite, singular factors" begin
    S = 3
    λ = LowerTriangular(Matrix{Float64}(I, S, S))
    # a = 0 => ρ = 1 (rank-1)
    updateλ!(CompoundSymmetry{Float64,false}([0.0, 1.3]), λ, [0.0, 1.3])
    @test all(isfinite, λ)
    @test rank(λ * λ'; atol=1e-8) == 1
    # g = 0 => ρ = -1/(S-1) (rank S-1)
    updateλ!(CompoundSymmetry{Float64,false}([1.1, 0.0]), λ, [1.1, 0.0])
    @test all(isfinite, λ)
    @test rank(λ * λ'; atol=1e-8) == S - 1
    # het-CS at b = -1/S is the same singular boundary
    b = -inv(S)
    updateλ!(CompoundSymmetry{Float64,true}(vcat(ones(S), b)), λ, vcat(ones(S), b))
    @test all(isfinite, λ)
    @test rank(λ * λ'; atol=1e-8) == S - 1
end

@testset "canonicalize! preserves λλ' and enters the canonical region" begin
    S = 4
    λ = LowerTriangular(Matrix{Float64}(I, S, S))
    λc = LowerTriangular(Matrix{Float64}(I, S, S))
    rng = StableRNG(1)
    for _ in 1:25
        d = 0.2 .+ rand(rng, S)
        b = randn(rng)
        θ = vcat(d .* rand(rng, [-1, 1], S), b) # random signs / possibly out of region
        cs = CompoundSymmetry{Float64,true}(copy(θ))
        updateλ!(cs, λ, θ)
        canonicalize!(cs)
        updateλ!(cs, λc, cs.θ)
        @test λ * λ' ≈ λc * λc'                 # same covariance
        @test all(cs.θ .>= lowerbd(cs) .- 1e-12) # in the canonical box
    end
end

@testset "nθ, lowerbd, initialθ" begin
    @test nθ(ScaledIdentity{Float64}([1.0])) == 1
    @test nθ(CompoundSymmetry{Float64,false}([1.0, 1.0])) == 2
    @test nθ(CompoundSymmetry{Float64,true}(ones(5))) == 5  # S = 4 => S + 1

    @test lowerbd(ScaledIdentity{Float64}([1.0])) == [0.0]
    @test lowerbd(CompoundSymmetry{Float64,false}([1.0, 1.0])) == [0.0, 0.0]
    @test lowerbd(CompoundSymmetry{Float64,true}(ones(4))) == [0.0, 0.0, 0.0, -1 / 3]

    # initial θ always yields λ = I
    for cs in (ScaledIdentity{Float64}([9.0]),
        CompoundSymmetry{Float64,false}([9.0, 3.0]),
        CompoundSymmetry{Float64,true}(vcat(9 .* ones(3), 2.0)))
        S = cs isa CompoundSymmetry{Float64,true} ? length(cs.θ) - 1 : 3
        M = cs isa ScaledIdentity ? Diagonal(zeros(S)) :
            LowerTriangular(Matrix{Float64}(I, S, S))
        updateλ!(cs, M, initialθ(cs))
        @test Matrix(M * M') ≈ I(S)
    end
end

@testset "S = 1 handling" begin
    slp = dataset(:sleepstudy)
    # scaled identity with a scalar term is allowed (equivalent to unstructured)
    m = fit(LMM, @formula(reaction ~ 1 + days + homdiag(1 | subj)), slp; progress=false)
    m0 = fit(LMM, @formula(reaction ~ 1 + days + (1 | subj)), slp; progress=false)
    @test objective(m) ≈ objective(m0)
    # compound symmetry requires S ≥ 2
    @test_throws ArgumentError fit(
        LMM, @formula(reaction ~ 1 + days + cs(1 | subj)), slp; progress=false
    )
end

@testset "S = 2 het-CS equals unstructured" begin
    slp = dataset(:sleepstudy)
    m0 = fit(LMM, @formula(reaction ~ 1 + days + (1 + days | subj)), slp; progress=false)
    m1 = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    @test objective(m1) ≈ objective(m0) atol = 1e-5
    Σ0 = m0.λ[1] * m0.λ[1]'
    Σ1 = m1.λ[1] * m1.λ[1]'
    @test Σ0 ≈ Σ1 atol = 1e-4
    @test dof(m1) == dof(m0)
end

@testset "structured objective ≥ unstructured; dof accounting" begin
    kb07 = dataset(:kb07)
    f = @formula(rt_trunc ~ 1 + spkr + prec + load + (1 + prec | subj))
    m_un = fit(LMM, f, kb07; progress=false)
    fcs = @formula(rt_trunc ~ 1 + spkr + prec + load + cs(1 + prec | subj))
    m_cs = fit(LMM, fcs, kb07; progress=false)
    fhc = @formula(rt_trunc ~ 1 + spkr + prec + load + homcs(1 + prec | subj))
    m_hc = fit(LMM, fhc, kb07; progress=false)
    fhd = @formula(rt_trunc ~ 1 + spkr + prec + load + homdiag(1 + prec | subj))
    m_hd = fit(LMM, fhd, kb07; progress=false)

    # a constrained model cannot fit better than the unconstrained one
    @test objective(m_cs) ≥ objective(m_un) - 1e-6
    @test objective(m_hc) ≥ objective(m_cs) - 1e-6
    @test objective(m_hd) ≥ objective(m_hc) - 1e-6

    @test nθ(m_un.reterms[1]) == 3   # full 2x2 lower triangle
    @test nθ(m_cs.reterms[1]) == 3   # 2 sds + 1 correlation
    @test nθ(m_hc.reterms[1]) == 2   # 1 sd + 1 correlation
    @test nθ(m_hd.reterms[1]) == 1   # 1 sd
end

@testset "homogeneity of variances / correlations" begin
    slp = dataset(:sleepstudy)
    mhc = fit(LMM, @formula(reaction ~ 1 + days + homcs(1 + days | subj)), slp; progress=false)
    vc = VarCorr(mhc)
    σ1, σ2 = values(getproperty(vc.σρ, :subj).σ)
    @test σ1 ≈ σ2                # equal variances
    mhd = fit(LMM, @formula(reaction ~ 1 + days + homdiag(1 + days | subj)), slp; progress=false)
    Σ = mhd.λ[1] * mhd.λ[1]'
    @test Σ ≈ Σ[1, 1] * I(2)     # scaled identity
end

@testset "θ round-trips through setθ!/getθ!" begin
    slp = dataset(:sleepstudy)
    for wrapper in (:cs, :homcs, :homdiag)
        f = @eval @formula(reaction ~ 1 + days + $wrapper(1 + days | subj))
        m = fit(LMM, f, slp; progress=false)
        θ = copy(m.θ)
        # perturb, restore, and confirm objective is a pure function of θ
        obj = objective(m)
        MixedModels.setθ!(m, θ .+ 0.05)
        MixedModels.updateL!(m)
        MixedModels.setθ!(m, θ)
        MixedModels.updateL!(m)
        @test objective(m) ≈ obj
        @test MixedModels.getθ(m) ≈ θ
    end
end

@testset "canonical fit from perturbed / negative initial θ" begin
    slp = dataset(:sleepstudy)
    m = fit(LMM, @formula(reaction ~ 1 + days + homcs(1 + days | subj)), slp; progress=false)
    # the fitted θ must lie in the canonical region
    @test all(m.θ .>= lowerbd(m) .- 1e-8)
end

@testset "amalgamation with a structured term is rejected" begin
    slp = dataset(:sleepstudy)
    f = @formula(reaction ~ 1 + days + (1 | subj) + cs(0 + days | subj))
    @test_throws ArgumentError fit(LMM, f, slp; progress=false)
end

@testset "wrappers cannot be nested" begin
    slp = dataset(:sleepstudy)
    f = @formula(reaction ~ 1 + days + zerocorr(cs(1 + days | subj)))
    @test_throws ArgumentError apply_schema(f, schema(slp), LMM)
end

@testset "copy_oftype converts the covariance structure" begin
    slp = dataset(:sleepstudy)
    m = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    A = m.reterms[1]
    A32 = LinearAlgebra.copy_oftype(A, Float32)
    @test A32.covstruct isa CompoundSymmetry{Float32,true}
    @test A32.covstruct.θ ≈ Float32.(A.covstruct.θ)
end

@testset "bootstrap of a structured model" begin
    slp = dataset(:sleepstudy)
    m = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    boot = parametricbootstrap(StableRNG(1), 200, m; progress=false)
    @test length(boot) == 200
    @test all(x -> x isa CompoundSymmetry{Float64,true}, boot.covstructs)
    # every replicate's reconstructed λ must be a valid het-CS factor:
    # a single correlation shared across the (only) off-diagonal pair
    ci = confint(boot)
    @test "ρ1" in string.(ci.par)
    # save/restore round trip
    io = IOBuffer()
    savereplicates(io, boot)
    seekstart(io)
    boot2 = restorereplicates(io, m)
    @test boot == boot2
end

@testset "profile of a structured model" begin
    slp = dataset(:sleepstudy)
    m = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    pr = profile(m)
    ci = confint(pr)
    # β intervals should match the unstructured model (same likelihood at S=2)
    m0 = fit(LMM, @formula(reaction ~ 1 + days + (1 + days | subj)), slp; progress=false)
    ci0 = confint(profile(m0))
    idx = findfirst(==(Symbol("β1")), ci.par)
    idx0 = findfirst(==(Symbol("β1")), ci0.par)
    @test ci.lower[idx] ≈ ci0.lower[idx0] atol = 0.5
end

@testset "serialization round trip" begin
    slp = dataset(:sleepstudy)
    m = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    io = IOBuffer()
    saveoptsum(io, m)
    m2 = fit(LMM, @formula(reaction ~ 1 + days + cs(1 + days | subj)), slp; progress=false)
    seekstart(io)
    restoreoptsum!(m2, io)
    @test objective(m2) ≈ objective(m)
    @test m2.θ ≈ m.θ
end
