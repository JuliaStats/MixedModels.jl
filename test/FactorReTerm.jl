using DataFrames, LinearAlgebra, MixedModels, Random, RData, SparseArrays, StatsModels, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

const LMM = LinearMixedModel

@testset "scalarReMat" begin
    ds = dat[:Dyestuff]
    f1 = @formula(Y ~ 1 + (1|G))
    y1, Xs1 = modelcols(apply_schema(f1, schema(ds), LMM), ds)
    sf = Xs1[2]
    psts = dat[:Pastes]
    f2 = @formula(Y ~ 1 + (1|G) + (1|H))
    y2, Xs2 = modelcols(apply_schema(f2, schema(psts), LMM), psts)
    sf1 = Xs2[2]
    sf2 = Xs2[3]

    @testset "size" begin
        @test size(sf) == (30, 6)
        @test size(sf,1) == 30
        @test size(sf,2) == 6
        @test size(sf,3) == 1
        @test size(sf1) == (60, 30)
        @test size(sf2) == (60, 10)
    end

    @testset "utilities" begin
        @test MixedModels.levs(sf) == string.('A':'F')
        @test MixedModels.nlevs(sf) == 6
        @test eltype(sf) == Float64
        @test sparse(sf) == sparse(1:30, sf.refs, ones(30))
        fsf = Matrix(sf)
        @test size(fsf) == (30, 6)
        @test count(!iszero, fsf) == 30
        @test sort!(unique(fsf)) == [0.0, 1.0]
        @test cond(sf) == 1.0
        @test MixedModels.nθ(sf) == 1
        @test MixedModels.getθ(sf) == ones(1)
        @test MixedModels.getθ!(Vector{Float64}(undef, 1), sf) == ones(1)
        @test lowerbd(sf) == zeros(1)
        @test MixedModels.getθ(setθ!(sf, [0.5])) == [0.5]
        MixedModels.unscaledre!(Vector{Float64}(undef, 30), sf)
        @test_throws DimensionMismatch MixedModels.getθ!(Float64[], sf)
        @test_throws DimensionMismatch setθ!(sf, ones(2))
    end

    @testset "products" begin
        @test ones(30, 1)'sf == fill(5.0, (1, 6))
        @test mul!(Array{Float64}(undef, (size(sf1, 2), size(sf2, 2))), sf1', sf2) == Array(sf1'sf2)

        crp = sf'sf
        @test isa(crp, Diagonal{Float64})
        crp1 = copy(crp)
        @test crp1 == crp
        @test crp[2,6] == 0
        @test crp[6,6] == 5
        @test size(crp) == (6,6)
        @test crp.diag == fill(5.,6)
        rhs = y1'sf
        @test rhs == reshape([7525.0,7640.0,7820.0,7490.0,8000.0,7350.0], (1, 6))
        @test ldiv!(crp, copy(rhs)') == [1505.,1528.,1564.,1498.,1600.,1470.]

        @test isa(sf1'sf1, Diagonal{Float64})
        @test isa(sf2'sf2, Diagonal{Float64})
        @test isa(sf2'sf1,SparseMatrixCSC{Float64})

        @test_broken lmul!(Λ(sf)', ones(6)) == fill(0.5, 6)
        @test_broken rmul!(ones(6, 6), Λ(sf)) == fill(0.5, (6, 6))

        @test_broken mul!(Matrix{Float64}(undef, 1, 6), Λ(sf), ones(1, 6)) == fill(0.5, (1,6))
    end

    @testset "reweight!" begin
        wts = rand(MersenneTwister(1234321), size(sf, 1))
        @test isapprox(vec(MixedModels.reweight!(sf, wts).wtz), wts)
    end
end

@testset "RandomEffectsTerm" begin
    slp = dat[:sleepstudy]
    contrasts =  Dict{Symbol,Any}()

    @testset "Detect same variable as blocking and experimental" begin
        f = @formula(Y ~ 1 + (1 + G|G))
        @test_throws ArgumentError apply_schema(f, schema(f, slp, contrasts), LinearMixedModel)
    end

    @testset "Detect both blocking and experimental variables" begin
        # note that U is not in the fixed effects because we want to make square
        # that we're detecting all the variables in the random effects
        f = @formula(Y ~ 1 + (1 + U|G))
        form = apply_schema(f, schema(f, slp, contrasts), LinearMixedModel)
        @test StatsModels.termvars(form.rhs) == [:U, :G]
    end
end

@testset "Categorical Blocking Variable" begin
    # deepcopy because we're going to modify it
    slp = deepcopy(dat[:sleepstudy])
    contrasts =  Dict{Symbol,Any}()
    f = @formula(Y ~ 1 + (1|G))

    # String blocking-variables work fine because StatsModels is smart enough to
    # treat strings to treat strings as Categorical. Note however that this is a
    # far less efficient to store the original dataframe, although it doesn't
    # matter for the contrast matrix
    slp[!,:G] = convert.(String, slp[!, :G])
    # @test_throws ArgumentError LinearMixedModel(f, slp)
    slp[!,:G] = parse.(Int, slp[!, :G])
    @test_throws ArgumentError LinearMixedModel(f, slp)
end

#=
@testset "vectorRe" begin
    slp = dat[:sleepstudy]
    corr = VectorFactorReTerm(slp[:G], vcat(ones(1, size(slp, 1)), slp[:U]'),
        :G, ["(Intercept)", "U"], [2])
    nocorr = VectorFactorReTerm(slp[:G], vcat(ones(1, size(slp, 1)), slp[:U]'),
            :G, ["(Intercept)", "U"], [1, 1])
    Reaction = slp[:Y]

    @testset "sizes" begin
        @test size(corr) == (180,36)
        @test size(nocorr) == (180,36)
    end

    @testset "utilities" begin
        @test MixedModels.levs(corr) == levels(slp[:G])
        @test MixedModels.nlevs(corr) == 18
        @test MixedModels.vsize(corr) == 2
        @test MixedModels.nrandomeff(corr) == 36
        @test eltype(corr) == Float64
        @test nnz(sparse(corr)) == 360
        @test cond(corr) == 1.0
        @test MixedModels.nθ(corr) == 3
        @test MixedModels.nθ(nocorr) == 2
        @test MixedModels.getθ(corr) == [1.0, 0.0, 1.0]
        @test MixedModels.getθ(nocorr) == ones(2)
        @test MixedModels.getθ!(Vector{Float64}(undef, 2), nocorr) == ones(2)
        @test lowerbd(nocorr) == zeros(2)
        @test lowerbd(corr) == [0.0, -Inf, 0.0]
        @test MixedModels.getθ(setθ!(corr, fill(0.5, 3))) == [0.5, 0.5, 0.5]
        @test_throws DimensionMismatch MixedModels.getθ!(Vector{Float64}(undef, 2), corr)
        @test_throws DimensionMismatch setθ!(corr, ones(2))
    end

    vrp = corr'corr

    @test isa(vrp, UniformBlockDiagonal{Float64})
    @test size(vrp) == (36, 36)

    @test mul!(Array{Float64}(undef, 36, 36), corr', corr) == Matrix(vrp)
    scl = ScalarFactorReTerm(slp[:G].refs, levels(slp[:G]), Array(slp[:U]), Array(slp[:U]), :G, ["U"], 1.0)

    @test sparse(corr)'sparse(scl) == corr'scl

    b = mul!(Matrix{Float64}(undef, 2,18), Λ(corr).data, ones(2,18))
    @test b == reshape(repeat([0.5, 1.0], outer=18), (2,18))

    @testset "reweight!" begin
        wts = rand(MersenneTwister(1234321), size(corr, 1))
        @test MixedModels.reweight!(corr, wts).wtz[1, :] == wts
        @test corr.z[1, :] == ones(size(corr, 1))
    end

end
=#
