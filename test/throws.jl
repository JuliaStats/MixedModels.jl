### Test error exits
f = Yield ~ 1 + (1 | Batch)
mf = ModelFrame(f, ds)
Λ = [LowerTriangular(eye(1)) for i = 1]
y = convert(Vector{Float64}, ds[:Yield])
trms = push!(Any[], remat(:(1|Batch), ds), ones(30, 1), reshape(y, (length(y), 1)))

@test_throws DomainError LinearMixedModel(f, mf, trms, Λ, fill(-1., 30))

@test_throws ArgumentError lmm(Yield ~ 1, ds)

modl = lmm(Yield ~ 1 + (1|Batch), ds);

@test_throws DimensionMismatch MixedModels.getθ!(Array(Float64, (2,)), modl)
@test_throws DimensionMismatch setθ!(modl, [0., 1.])

@test_throws DimensionMismatch LinAlg.Ac_ldiv_B!(UpperTriangular(fm3.R[1, 1]), ones(30, 1))

const tri = LowerTriangular(eye(3))
const hblk = MixedModels.HBlkDiag(ones(2,3,2))
const dd3 = Diagonal(ones(3))

@test_throws DimensionMismatch setθ!(tri, ones(2))
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
@test_throws DimensionMismatch LinAlg.A_ldiv_B!(dd3, Diagonal(ones(2)))

const speye4 = speye(4)
@test_throws ArgumentError MixedModels.inject!(speye(4), sparse(ones(4,4)))
@test_throws DimensionMismatch LinAlg.A_ldiv_B!(dd3, speye4)
@test_throws DimensionMismatch LinAlg.Ac_ldiv_B!(UpperTriangular(hblk), ones(1,1))
@test_throws DimensionMismatch MixedModels.unscaledre!(ones(1), modl.wttrms[1], ones(1, 6))

@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye4, ones(4,1))
@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye4, speye4)
@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye(3), speye4)
@test_throws DimensionMismatch MixedModels.downdate!(ones(1,1), speye4)

@test_throws DimensionMismatch MixedModels.inject!(speye4, speye(3))

@test_throws DimensionMismatch MixedModels.unscaledre!(zeros(size(slp, 1)),
  remat(:(1 + Days | Subject), slp), ones(1,30))

@test_throws DimensionMismatch MixedModels.tscale!(LowerTriangular(eye(2)), Diagonal(ones(30)))
@test_throws DimensionMismatch MixedModels.tscale!(Diagonal(ones(30)), LowerTriangular(eye(2)))
@test_throws ArgumentError MixedModels.lrt(modl)
