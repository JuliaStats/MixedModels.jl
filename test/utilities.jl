using LinearAlgebra, MixedModels, Random, SparseArrays, Test

@testset "utilities" begin
	@test MixedModels.average(1.1, 1.2) == 1.15
	@test MixedModels.densify(sparse(1:5, 1:5, ones(5))) == Diagonal(ones(5))
	rsparsev = SparseVector(float.(rand(MersenneTwister(123454321), Bool, 20)))
	@test MixedModels.densify(rsparsev) == Vector(rsparsev)
	@test MixedModels.densify(Diagonal(rsparsev)) == Diagonal(Vector(rsparsev))
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
