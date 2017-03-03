"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return the value of `log(det(triu(A)))` calculated in place.
"""
LD{T<:Number}(d::Diagonal{T}) = sum(log, d.diag)

function LD{T}(d::Diagonal{LowerTriangular{T, Matrix{T}}})
    s = log(one(T))
    for dd in d.diag, i in diagind(dd)
        s += log(dd[i])
    end
    s
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` calculated in place.
"""
logdet(m::LinearMixedModel) = 2 * sum(LD, view(diag(m.L.data.blocks), 1:length(m.Λ)))
