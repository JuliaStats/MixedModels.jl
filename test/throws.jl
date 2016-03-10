### Test error exits
mf = ModelFrame(Yield ~ 1 + (1|Batch), ds)
Λ = [LowerTriangular(eye(1)) for i = 1]
y = convert(Vector,ds[:Yield])
Rem = push!([],remat(:(1|Batch), ds))
X = ones(30,1)

@test_throws ArgumentError LinearMixedModel(mf, ones(1), Λ, X, y, Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf, Rem, Λ, X, y[1:20], Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf, Rem, Λ, X[1:20,:], y, Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf, append!(copy(Rem), Rem), Λ, X, y, Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf, Rem, append!(copy(Λ), Λ), X, y, Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,Rem, [LowerTriangular(eye(2)) for i in 1], X, y, Float64[])
@test_throws ArgumentError LinearMixedModel(mf, Rem, Λ, X, y, fill(-1., 30))

@test_throws ArgumentError lmm(Yield ~ 1,ds)

modl = lmm(Yield ~ 1 + (1|Batch), ds);

@test_throws ArgumentError modl[:ϕ] = [1.]
@test_throws DimensionMismatch modl[:θ] = [0.,1.]

const tri = LowerTriangular(eye(3))
const hblk = MixedModels.HBlkDiag(ones(2,3,2))
const dd3 = Diagonal(ones(3))
@test_throws KeyError tri[:foo]
@test_throws KeyError tri[:foo] = ones(3)
@test_throws DimensionMismatch tri[:θ] = ones(2)
@test_throws DimensionMismatch MixedModels.tscale!(tri, ones(2))
@test_throws DimensionMismatch MixedModels.tscale!(ones(2,2), tri)
@test_throws DimensionMismatch MixedModels.tscale!(tri, hblk)
@test_throws ArgumentError Base.cholfact!(hblk)
@test_throws DimensionMismatch MixedModels.downdate!(dd3, Diagonal(ones(2)))
@test_throws DimensionMismatch MixedModels.downdate!(dd3, sparse(ones(4,1)))
@test_throws DimensionMismatch MixedModels.downdate!(ones(2,2), dd3, ones(3,3))
@test_throws DimensionMismatch MixedModels.downdate!(ones(2,2), dd3, ones(2,2))
@test_throws DimensionMismatch MixedModels.inject!(ones(2,2), dd3)
@test_throws DimensionMismatch MixedModels.inject!(ones(3,2), dd3)
@test_throws DimensionMismatch Base.LinAlg.A_ldiv_B!(dd3, Diagonal(ones(2)))

const speye4 = speye(4)
@test_throws DimensionMismatch Base.LinAlg.A_ldiv_B!(dd3, speye4)
@test_throws DimensionMismatch Base.LinAlg.Ac_ldiv_B!(UpperTriangular(hblk), ones(1,1))
@test_throws DimensionMismatch MixedModels.unscaledre!(ones(1), modl.trms[1], modl.Λ[1], ones(1, 6))

@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye(4), ones(4,1))

@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye(3), speye(4))
@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye(4))
