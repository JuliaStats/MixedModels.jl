"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::LowerTriangular{T, UniformBlockDiagonal{T}}) where {T}
    s = log(one(T))
#= FIXME This is broken
    dind = diagind(K, K)
    for dd in d.data.data, i in dind
        s += log(dd[i])
    end
=#
    s
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` evaluated in place.
"""
logdet(m::LinearMixedModel{T}) where {T} = 2sum(i -> T(LD(m.L.data[Block(i, i)])), 1:nreterms(m))

logdet(m::GeneralizedLinearMixedModel) = logdet(m.LMM)
