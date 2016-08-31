using MixedModels,Base.Test
const m1 = LowerTriangular(eye(3))

@test size(m1) == (3,3)
@test size(m1,1) == 3
@test size(m1,2) == 3
@test size(m1,3) == 1
const m1c = copy(m1)
@test MixedModels.nlower(m1) == 6
@test m1c == m1
@test getθ(m1) == [1., 0, 0, 1, 0, 1]
@test lowerbd(m1) == [0., -Inf, -Inf, 0., -Inf, 0.]

setθ!(m1, collect(1.:6))                  # assignment of parameters
@test getθ(m1) == collect(1.:6)           # check assignment and extraction
@test m1c != m1
@test copy!(m1c,m1) == m1

@test full(m1) == reshape([1.,2,3,0,4,5,0,0,6],(3,3))
