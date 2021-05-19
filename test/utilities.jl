using LinearAlgebra
using MixedModels
using StableRNGs
using SparseArrays
using Test

using MixedModels: allequal, average, densify, dataset

@isdefined(io) || const global io = IOBuffer()
include("modelcache.jl")

@testset "average" begin
	@test average(1.1, 1.2) == 1.15
end

@testset "densify" begin
	@test densify(sparse(1:5, 1:5, ones(5))) == Diagonal(ones(5))
	rsparsev = SparseVector(float.(rand(StableRNG(123454321), Bool, 20)))
	@test densify(rsparsev) == Vector(rsparsev)
	@test densify(Diagonal(rsparsev)) == Diagonal(Vector(rsparsev))
end

@testset "allequal" begin
	@test allequal((true, true, true))
	@test allequal([true, true, true])
	@test !allequal((true, false, true))
	@test !allequal([true, false, true])
	@test !allequal(collect(1:4))
	@test allequal((false, false, false))
	@test allequal([false, false, false])
	@test allequal(ones(3))
	@test allequal(1, 1, 1)

	# equality of arrays with broadcasting
	@test allequal(["(Intercept)", "days"], ["(Intercept)", "days"])
end

@testset "threaded_replicate" begin
	rng = StableRNG(42);
	single_thread = replicate(10;use_threads=false) do; only(randn(rng, 1)) ; end
	rng = StableRNG(42);
	multi_thread = replicate(10;use_threads=true) do
		if Threads.threadid() % 2 == 0
			sleep(0.001)
		end
		r = only(randn(rng, 1));
	end

	@test all(sort!(single_thread) .≈ sort!(multi_thread))
end

@testset "datasets" begin
	@test isa(MixedModels.datasets(), Vector{String})
	@test length(MixedModels.dataset(:dyestuff)) == 2
	@test length(MixedModels.dataset("dyestuff")) == 2
	dyestuff = MixedModels.dataset(:dyestuff);
	@test keys(dyestuff) == [:batch, :yield]
	@test length(dyestuff.batch) == 30
	@test_throws ArgumentError MixedModels.dataset(:foo)
end

@testset "PCA" begin
	io = IOBuffer()
	pca = models(:kb07)[3].PCA.item

	show(io, pca, covcor=true, loadings=false)
	str = String(take!(io))
	@test !isempty(findall("load: yes", str))

	show(io, pca, covcor=false, loadings=true)
	str = String(take!(io))
	@test !isempty(findall("PC1", str))
	@test !isempty(findall("load: yes", str))
end
