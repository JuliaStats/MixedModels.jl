using LinearAlgebra
using MixedModels
using MixedModelsDatasets: dataset

"""
    grad_comp(m::LinearMixedModel)

Returns the gradient of the log-determinant part of the objective for `m`,
which must have a single, vector-valued random-effects term.
"""
function grad_comp(m::LinearMixedModel{T}) where {T}
    (; reterms, parmap, A, L) = m
    A11 = first(A).data
    L11 = first(L).data
    λ = only(reterms).λ                 # checks that there is exactly one random-effects term
    λdot = similar(λ)
    face = similar(λ.data)
    grad = zeros(T, length(parmap))
    for (p, pm) in enumerate(parmap)
        fill!(λdot, zero(T))
        λdot[pm[2], pm[3]] = one(T)
        for k in axes(A11, 3)           # loop over faces of A[1].data
            rmul!(lmul!(λ', copyto!(face, view(A11, :, :, k))), λdot)
            for i in axes(face, 1)      # symmetrize the face and double the diagonal
                for j in 1:(i - 1)
                    ijsum = face[i, j] + face[j, i]
                    face[j, i] = face[i, j] = ijsum
                end
                face[i, i] *= 2
            end
            Lface = LowerTriangular(view(L11, :, :, k))
            rdiv!(ldiv!(Lface, face), Lface')
            for i in diagind(face)
                grad[p] += face[i]
            end
        end 
    end
    return grad
end