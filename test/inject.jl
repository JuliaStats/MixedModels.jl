const ut3 = UpperTriangular(eye(3))
const ut2 = UpperTriangular(eye(2))

@test_throws DimensionMismatch MixedModels.inject!(ut3,ut2)
@test MixedModels.inject!(UpperTriangular(zeros(3,3)),ut3) == eye(3)

speye41 = sparse(vcat(1:4, 1), vcat(1:4, 2), ones(5))
@test nonzeros(MixedModels.inject!(speye41, speye4)) == [1., 0, 1, 1, 1]
