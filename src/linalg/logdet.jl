"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::UniformBlockDiagonal{T}) where T
    m, n, k = size(d.data)
    dind = diagind(m, n)
    sum(log, f[i] for f in d.facevec, i in dind)
end

LD(d::DenseMatrix{T}) where {T} = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` evaluated in place.
"""
function logdet(m::LinearMixedModel{T}) where {T}
    s = zero(T)
    for (i, trm) in enumerate(m.trms)
        if isa(trm, AbstractFactorReTerm)
            s += T(LD(m.L.data[Block(i, i)]))
        end
    end
    2s
end

logdet(m::GeneralizedLinearMixedModel) = logdet(m.LMM)
