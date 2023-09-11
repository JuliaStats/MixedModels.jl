using LinearAlgebra
using MixedModels
using StableRNGs
using SparseArrays
using Test

using MixedModels: isconstant, average, densify, dataset
using StatsModels: FormulaTerm

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

@testset "isconstant" begin
	@test isconstant((true, true, true))
	@test isconstant([true, true, true])
	@test !isconstant((true, false, true))
	@test !isconstant([true, false, true])
	@test !isconstant(collect(1:4))
	@test isconstant((false, false, false))
	@test isconstant([false, false, false])
	@test isconstant(ones(3))
	@test isconstant(1, 1, 1)
	# equality of arrays with broadcasting
	@test isconstant(["(Intercept)", "days"], ["(Intercept)", "days"])
	# arrays or tuples with missing values
	@test !isconstant([missing, 1])
	@test isconstant(Int[])
	@test isconstant(Union{Int,Missing}[missing, missing, missing])
end

@testset "replicate" begin
	@test_logs (:warn, r"use_threads is deprecated") replicate(string, 1; use_threads=true)
	@test_logs (:warn, r"hide_progress") replicate(string, 1; hide_progress=true)
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

@testset "formula" begin
	@test formula(first(models(:sleepstudy))) isa FormulaTerm
	@test formula(first(models(:contra))) isa FormulaTerm
end
