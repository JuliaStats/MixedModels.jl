## Utilities

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

Base.cholfact(s::Symmetric{Float64}) = cholfact(symcontents(s), symbol(s.uplo))

function scaleinv!(b::StridedVector,sc::StridedVector)
    (n = length(b)) == length(sc) || throw(DimensionMismatch(""))
    @inbounds for i in 1:n
        b[i] /= sc[i]
    end
    b
end

symcontents(s::Symmetric) = VERSION ≥ v"0.4-" ? s.data : s.S
