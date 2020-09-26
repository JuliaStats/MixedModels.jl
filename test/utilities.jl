using LinearAlgebra
using MixedModels
using Random
using SparseArrays
using Test

using MixedModels: allequal, average, densify, dataset

const io = IOBuffer()
include("modelcache.jl")

@testset "average" begin
	@test average(1.1, 1.2) == 1.15
end

@testset "densify" begin
	@test densify(sparse(1:5, 1:5, ones(5))) == Diagonal(ones(5))
	rsparsev = SparseVector(float.(rand(MersenneTwister(123454321), Bool, 20)))
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
	rng = MersenneTwister(42);
	single_thread = replicate(10,use_threads=false) do; randn(rng, 1)[1] ; end
	rng = MersenneTwister(42);
	multi_thread = replicate(10,use_threads=true) do
		if Threads.threadid() % 2 == 0
			sleep(0.001)
		end
		r = randn(rng, 1)[1];
	end

	@test all(sort!(single_thread) .== sort!(multi_thread))
end

@testset "datasets" begin
	@test isa(MixedModels.datasets(), Vector{String})
	@test size(MixedModels.dataset(:dyestuff)) == (30, 2)
	@test size(MixedModels.dataset("dyestuff")) == (30, 2)
	@test_throws ArgumentError MixedModels.dataset(:foo)
end

@testset "PCA" begin
	pca = models(:kb07)[3].PCA.item
	
	show(io, pca, covcor=true, loadings=false)
	str = String(take!(io))
	@test !isempty(findall("load: yes        0.18   0.16  -0.14   1.0", str))
	
	show(io, pca, covcor=false, loadings=true)
	str = String(take!(io))
	@test !isempty(findall("PC1    PC2    PC3    PC4", str))
	@test !isempty(findall("load: yes       -0.19  -0.63  -0.75  -0.02", str))
end