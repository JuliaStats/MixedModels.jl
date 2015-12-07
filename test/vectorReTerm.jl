const Days = convert(Vector{Float64},slp[:Days])
const vf = VectorReMat(slp[:Subject],hcat(ones(length(Days)),Days)',:Subject,["(Intercept)","Days"])
const Reaction = convert(Array,slp[:Reaction])

@test size(vf) == (180,36)
const vrp = vf'vf
@test (vf'ones(size(vf,1)))[1:4] == [10.,45,10,45]
@test isa(vrp,MixedModels.HBlkDiag{Float64})
@test eltype(vrp) == Float64
@test size(vrp) == (36,36)
const rhs1 = ones(36,2)
const x = similar(rhs1)
const b1 = copy(vrp.arr[:,:,1]) + I
@test sub(MixedModels.inflate!(vrp).arr,:,:,1) == b1
const cf = cholfact(b1)
@test triu!(sub(cholfact!(vrp).arr,:,:,1)) == cf[:U]
