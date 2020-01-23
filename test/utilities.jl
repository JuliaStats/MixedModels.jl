using LinearAlgebra, MixedModels, Random, SparseArrays, Test

@testset "utilities" begin
	@test MixedModels.average(1.1, 1.2) == 1.15
	@test MixedModels.densify(sparse(1:5, 1:5, ones(5))) == Diagonal(ones(5))
	rsparsev = SparseVector(float.(rand(MersenneTwister(123454321), Bool, 20)))
	@test MixedModels.densify(rsparsev) == Vector(rsparsev)
	@test MixedModels.densify(Diagonal(rsparsev)) == Diagonal(Vector(rsparsev))
end
