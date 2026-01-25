# Evaluate analytic gradient of the objective for ML or REML fitting of a LinearMixedModel

"""
  Omega_dot_diag_block!(blk, m::LinearMixedModel, p::Integer)

Fill `blk` as the non-zero diagonal block of ∂Ω/∂θₚ for parameter number `p` of model `m`.

For any `p` only one diagonal block of ∂Ω/∂θₚ will be non-zero.
"""
function Omega_dot_diag_block!(
  blk::Diagonal{T, Vector{T}},
  m::LinearMixedModel{T},
  p::Integer,
) where {T}  
  (; parmap, A, reterms) = m
  b, i, j = parmap[p]
  isone(i) && isone(j) || throw(ArgumentError("parameter $p should be from a scalar r.e. term"))
  # It is common for 'b' to be one as well but nested models can result in diagonal blk for b > 1
  blk_diag = blk.diag
  A_diag = A[kp1choose2(b)].diag        # will throw an error if the A[b,b] block is not Diagonal
  length(blk_diag) == length(A_diag) || throw(DimensionMismatch("A_diag and blk_diag have different lengths"))
  λ = only(reterms[b].λ)           # will throw an error if reterms[b] is not of size (1,1)
  for k in eachindex(blk_diag, A_diag)
    blk_diag[k] = T(2) * λ * A_diag[k]
  end       
  return blk
end

function Omega_dot_diag_block!(
  blk::Matrix{T},
  m::LinearMixedModel{T},
  p::Integer,
) where {T}  
  (; parmap, A, reterms) = m
  b, i, j = parmap[p]
  k = size(reterms[b].λ, 1)
  if isone(k)
    isone(i) && isone(j) || throw(ArgumentError("parameter $p should be from a scalar r.e. term"))
    A_diag = A[kp1choose2(b)].diag      # will throw an error if the A[b,b] block is not Diagonal
    size(blk, 1) == size(blk, 2) == length(A_diag) || throw(DimensionMismatch("A_diag and blk_diag have different lengths"))
    λ = only(reterms[b].λ)              # will throw an error if reterms[b].λ is not of size (1,1)
    _zero_out(blk)
    for k in eachindex(A_diag)
        blk[k, k] = T(2) * λ * A_diag[k]
    end       
    return blk
  end
  throw(ArgumentError("Code not yet written for k > 1"))
end

function Omega_dot_diag_block!(
  blk::UniformBlockDiagonal{T},
  m::LinearMixedModel{T},
  p::Integer
) where {T}
  (; parmap, A, reterms) = m
  b, i, j = parmap[p]
  Ablk = A[kp1choose2(b)]
  if !isa(Ablk, UniformBlockDiagonal{T})
    throw(ArgumentError("parmap[p] = $(parmap[p]) but A[$(kp1choose2(b))] is not UniformBlockDiagonal"))
  end
  blk_dat = fill!(blk.data, zero(T))
  Ablk_dat = Ablk.data
  λ = reterms[b].λ
  for k in axes(blk_dat, 3)
      # right multiply by λ-dot-transpose, which is zeros except for a single 1 at the i'th column and j'th row
      # thus we copy the i'th column of the k'th face of Ablk_dat into the j'th column of the k'th face of blk_dat
    copyto!(view(blk_dat, :, i, k), view(Ablk_dat, :, j, k))
    lmul!(λ, view(blk_dat, :, :, k)) # left-multiply by λ
    for jj in axes(λ, 2)             # symmetrize the face while multiplying the diagonal by 2
      for ii in 1:(jj - 1)
        val = blk_dat[ii, jj, k] + blk_dat[jj, ii, k]
        blk_dat[ii, jj, k] = blk_dat[jj, ii, k] = val
      end
      blk_dat[jj, jj, k] *= T(2)
    end
  end
  return blk
end

function LinearAlgebra.ldiv!(
    A::LowerTriangular{T,UniformBlockDiagonal{T}},
    B::UniformBlockDiagonal{T},
) where {T}
    A_dat = A.data.data
    B_dat = B.data
    if size(A_dat) ≠ size(B_dat)
        throw(DimensionMismatch("size(A_dat) = $(size(A_dat)) ≠ $(size(B_dat)) = size(B_dat)"))
    end
    for k in axes(B_dat, 3)
        ldiv!(LowerTriangular(view(A_dat, :, :, k)), view(B_dat, :, :, k))
    end
    return B
end

function LinearAlgebra.ldiv!(
    A::LowerTriangular{T,UniformBlockDiagonal{T}},
    B::Matrix{T},
) where {T}
    if size(A, 2) ≠ size(B, 1)
        throw(DimensionMismatch("size(A,2) = $(size(A,2)) ≠ $(size(B,1)) = size(B,1"))
    end
    A_dat = A.data.data
    axis1 = axes(A_dat, 1)
    offset = 0
    for k in axes(A_dat, 3)
        ldiv!(LowerTriangular(view(A_dat, :, :, k)), view(B, offset .+ axis1, :))
        offset += length(axis1)
    end
    return B
end

function LinearAlgebra.rdiv!(
    A::UniformBlockDiagonal{T},
    B::UpperTriangular{T, LinearAlgebra.Transpose{T, UniformBlockDiagonal{T}}},
) where {T}
    A_dat = A.data
    B_dat = B.data.parent.data
    if size(A_dat) ≠ size(B_dat)
        throw(DimensionMismatch("size(A_dat) = $(size(A_dat)) ≠ $(size(B_dat)) = size(B_dat)"))
    end
    for k in axes(A_dat, 3)
        rdiv!(view(A_dat, :, :, k), LowerTriangular(view(B_dat, :, :, k))')
    end
    return B
end

function Base.similar(A::UniformBlockDiagonal)
    return UniformBlockDiagonal(similar(A.data))
end

"""
    grad_blocks(m::LinearMixedModel{T})

Return Matrix{AbstractMatrix{T}} containing the gradient-evaluation blocks for model `m`.
"""
function grad_blocks(m::LinearMixedModel{T}) where {T}
    (; L, reterms) = m
    k = length(reterms) + 1
    val = sizehint!(AbstractMatrix{T}[], abs2(k))
    for j in 1:k
        for i in 1:k
            push!(val, similar(i ≥ j ? L[block(i, j)] : L[block(j, i)]'))
        end
    end
    return reshape(val, (k, k))
end

function _zero_out(A::Matrix{T}) where {T}
    return fill!(A, zero(T))
end

function _zero_out(A::Diagonal{T,Vector{T}}) where {T}
    fill!(A.diag, zero(T))
    return A
end

function _zero_out(A::UniformBlockDiagonal{T}) where {T}
    fill!(A.data, zero(T))
    return A
end

"""
    copyskip!(B::Matrix{T}, A::Matrix{T}, i::Integer, j::Integer, k::Integer) where {T}

Create `A * Ω_dot` in `B` where `Ω_dot` is the indicator for the `i`'th row and `j`'th column in a matrix of size `k`
"""
function copyskip!(B::Matrix{T}, A::Matrix{T}, i::Integer, j::Integer, k::Integer) where {T}
    m, n = size(A)
    (m, n) == size(B) || throw(DimensionMismatch("size(A) = $(size(A)) ≠ $(size(B)) = size(B)"))
    isone(k) && return copyto!(B, A)

    fill!(B, zero(T))
    q, r = divrem(n, k)
    iszero(r) || throw(DimensionMismatch("n = $n is not a multiple of k = $k"))
    offset = 0
    for _ in 1:q
        copyto!(view(B, :, offset + i), view(A, :, offset + j))
        offset += k      
    end    
    return B
end

function copyskip!(B::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, i::Integer, j::Integer, k::Integer) where {T}
    (A.m == B.m && A.n == B.n) || throw(DimensionMismatch("size(A) = $(size(A)) ≠ $(size(B)) = size(B)"))
    if any(A.colptr .≠ B.colptr) || any(rowvals(A) .≠ rowvals(B))
        throw(ArgumentError("A and B must have the same sparsity pattern"))
    end
    isone(k) && return copyto!(B, A)

    fill!(B.nzval, zero(T))
    q, r = divrem(A.n, k)
    iszero(r) || throw(DimensionMismatch("n = $n is not a multiple of k = $k"))
    rvB = rowvals(B)
    rvA = rowvals(A)
    nzB = nonzeros(B)
    nzA = nonzeros(A)
    offset = 0
    for _ in 1:q
        nzrB = nzrange(B, offset + i)
        nzrA = nzrange(A, offset + j)
        if !(view(rvB, nzrB) .== view(rvA, nzrA))
            throw(ArgumentError("A and B must have same sparsity pattern after shifting"))
        end
        copyto!(view(nzB, nzrB), view(nzA, nzrA))
        offset += k      
    end    
    return B
end

"""
    initialize_blocks!(blks::Matrix{AbstractMatrix{T}}, m::LinearMixedModel{T}, p::Integer)

Initialize the grad evaluation blocks, `blks`, for model `m`, for parameter `p`
"""
function initialize_blocks!(
    blks::Matrix{AbstractMatrix{T}},
    m::LinearMixedModel{T},
    p::Integer,
) where {T}
    (; parmap, A, reterms) = m
    b, i, j = parmap[p]
    k = size(reterms[b].λ, 1)
    Omega_dot_diag_block!(blks[b, b], m, p) # populate the b'th diagonal block
    for r in axes(blks, 1)                  # iterate over the lower triangle, transpose-copying to upper triangle
        if r ≠ b
            for c in 1:r
                if c ≠ b
                    _zero_out(blks[r, c])
                    _zero_out(blks[c, r])
                else
                    copyskip!(blks[r, c], A[block(r, c)], i, j, k)
                    copyto!(blks[c, r], blks[r, c]')
                end
            end
        else
            for c in 1:(r - 1)
                copyskip!(blks[r, c], A[block(r, c)], i, j, k)
                copyto!(blks[c, r], blks[r, c]')
            end
        end
    end
    return blks
end

"""
    eval_grad_p!(blks, m, p)

Evaluate the gradient component for parameter `p` in model `m` using blocks in `blks` for storage
"""
function eval_grad_p!(blks::Matrix{AbstractMatrix{T}}, m::LinearMixedModel{T}, p::Integer) where {T}
    L = m.L
#    b = first(parmap[p])                    # block, row and column for parameter p
    initialize_blocks!(blks, m, p)    # change this to pass b, i, j, k separately
    for kk in axes(blks, 2)                  # ldiv!(LowerTriangular(L), blks)
#        if jj ≥ b                     # maybe hold off on this at the expense of some multiplications by zero
        L11 = L[block(1, 1)]
        isa(L11, Diagonal) || (L11 = LowerTriangular(L11))
        C1 = ldiv!(L11, blks[1, kk])
        for ii in axes(blks, 1)[2:end]
            mul!(blks[ii, kk], L[block(ii, 1)], C1, -one(T), one(T))
        end
#        end
        for jj in axes(blks, 1)[2:end]
            Cj = ldiv!(LowerTriangular(L[block(jj, jj)]), blks[jj, kk])
            for ii in jj+1:lastindex(blks, 1)
                mul!(blks[ii, kk], L[block(ii, jj)], Cj, -one(T), one(T))
            end
        end
    end     
                # for k in axes(B,2)
                #     a11 = A[1,1]
                #     iszero(a11) && throw(SingularException(1))
                #     C1 = C[1,k] = a11 \ B[1,k]
                #     # fill C-column
                #     for i in axes(B,1)[2:end]
                #         C[i,k] = oA \ B[i,k] - _ustrip(A[i,1]) * C1
                #     end
                #     for j in axes(B,1)[2:end]
                #         ajj = A[j,j]
                #         iszero(ajj) && throw(SingularException(j))
                #         Cj = C[j,k] = _ustrip(ajj) \ C[j,k]
                #         for i in j+1:lastindex(A,1)
                #             C[i,k] -= _ustrip(A[i,j]) * Cj
                #         end
                #     end
                # end
    for ii in axes(blks, 1)                  # rdiv!(blks, transpose(LowerTriangular(L)))
        for jj in axes(blks, 2)
            blkij = blks[ii, jj]
            for kk in 1:(jj - 1)
                mul!(blkij, blks[ii, kk], transpose(L[block(jj, kk)]), -one(T), one(T))
            end
            rdiv!(blkij, transpose(LowerTriangular(L[block(jj, jj)])))
        end
    end
            # for i in axes(A,1)
            #     for j in axes(A,2)
            #         Aij = A[i,j]
            #         for k in firstindex(B,2):j - 1
            #             Aij -= C[i,k]*tfun(B[j,k])
            #         end
            #         unit || (iszero(B[j,j]) && throw(SingularException(j)))
            #         C[i,j] = Aij / (unit ? oB : tfun(B[j,j]))
            #     end
            # end                     
    return blks
end


