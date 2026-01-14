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
  isone(i) && isone(j) || throw(ArgumentError("parameter $b should be from a scalar r.e. term"))
  blk_diag = blk.diag
  A_diag = A[kp1choose2(b)].diag   # will throw an error if the A[b,b] block is not Diagonal
  length(blk_diag) == length(A_diag) || throw(DimensionMismatch("A_diag and blk_diag have different lengths"))
  λ = only(reterms[b].λ)           # will throw an error if reterms[b] is not of size (1,1)
  for k in eachindex(blk_diag, A_diag)
    blk_diag[k] = T(2) * λ * A_diag[k]
  end       
  return blk
end

function Omega_dot_diag_block!(
  blk::UniformBlockDiagonal{T},
  m::LinearMixedModel{T},
  p::Integer,
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
