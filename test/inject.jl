const ut3 = UpperTriangular(eye(3))
const ut2 = UpperTriangular(eye(2))

@test_throws DimensionMismatch MixedModels.inject!(ut3,ut2)
@test MixedModels.inject!(UpperTriangular(zeros(3,3)),ut3) ≈ eye(3)
