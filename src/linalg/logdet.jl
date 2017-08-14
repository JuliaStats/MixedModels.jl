"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::Diagonal{LowerTriangular{T, Matrix{T}}}) where T
    s = log(one(T))
    for dd in d.diag, i in diagind(dd)
        s += log(dd[i])
    end
    s
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` evaluated in place.
"""
function logdet(m::LinearMixedModel{T}) where T
    blks = m.L.data.blocks
    s = zero(T)
    for (k, t) in zip(diagind(blks), m.trms)
        if !isa(t, MatrixTerm)
            s += LD(blks[k])
        end
    end
    2s
end

logdet(m::GeneralizedLinearMixedModel{T}) where {T} = logdet(m.LMM)
