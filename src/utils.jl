## Utilities

## convert a lower Cholesky factor to a correlation matrix
function chol2cor(c::Matrix{Float64})
    m = Base.LinAlg.chksquare(c)
    m == 1 && return ones(1,1)
    std = broadcast(/, c, rowlengths(c))
    std * std'
end
chol2cor(t::Triangular) = chol2cor(full(t))
function chol2cor(c::Cholesky)
  c.uplo == 'L' || error("Code for upper Cholesky factor not yet written")
  (m = size(c.UL,1)) == 1 && return ones(eltype(c.UL),(1,1))
  t = full(c[:L])
  std = broadcast(/, t, rowlengths(t))
  std * std'
end
chol2cor(p::PDMat) = chol2cor(p.chol
                              )
## extract the lower triangle (with diagonal) of a matrix as a vector
function ltri(m::Matrix)
    n = Base.LinAlg.chksquare(m)
    res = Array(eltype(m), n*(n+1)>>1)
    pos = 0
    for j in 1:n, i in j:n
        res[pos += 1] = m[i,j]
    end
    res
end

## rowlengths(m) -> v : return a vector of the row lengths
rowlengths(m::Matrix{Float64}) = [norm(view(m,i,:))::Float64 for i in 1:size(m,1)]
rowlengths(t::Triangular{Float64}) = rowlengths(full(t))
rowlengths(p::PDMat) = rowlengths(p.chol[:L])

## Return a block in the Zt matrix from one term.
function ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nr*nblk                        # number of rows in result
    cf = length(m) < typemax(Int32) ? int32 : int64 # conversion function
    SparseMatrixCSC(NR,nc,
                    cf(cumsum(vcat(1,fill(nr,(nc,))))), # colptr
                    cf(vec(reshape([1:NR],(nr,int(nblk)))[:,v])), # rowval
                    vec(m))            # nzval
end
ztblk(m::Matrix,v::PooledDataVector) = ztblk(m,v.refs)

## copy contents of u, a vector of matrices, to cu, a vector, or vice versa
function copyucu!(cu::Vector{Float64},u::Vector,cu2u::Bool=true)
    pos = 0
    for ui in u
        ll = length(ui)
        cui = view(cu,pos+(1:ll))
        cu2u ? copy!(ui,cui) : copy!(cui,ui)
        pos += ll
    end
    cu2u ? u : cu
end
