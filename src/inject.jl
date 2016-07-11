"""
    inject!{T <: AbstractMatrix, S <: AbstractMatrix}(d::T, s::S)
Behaves like `copy!(d, s)` allowing for heterogeneous matrix types.
"""
inject!(d,s) = copy!(d,s)               # fallback method

function inject!(d::UpperTriangular, s::UpperTriangular)
    if (n = size(s, 2)) ≠ size(d, 2)
        throw(DimensionMismatch("size(s, 2) ≠ size(d, 2)"))
    end
    for j in 1:n
        inject!(Compat.view(d, 1 : j, j), Compat.view(s, 1 : j, j))
    end
    d
end

function inject!{T<:Real}(d::StridedMatrix{T}, s::Diagonal{T})
    sd = s.diag
    if length(sd) ≠ Compat.LinAlg.checksquare(d)  # why does d have to be square?
        throw(DimensionMismatch("size(d ,2) ≠ size(s, 2)"))
    end
    fill!(d, 0)
    @inbounds for i in eachindex(sd)
        d[i, i] = sd[i]
    end
    d
end

function inject!(d::Diagonal{Float64}, s::Diagonal{Float64})
    copy!(d.diag, s.diag)
    d
end

function inject!(d::SparseMatrixCSC{Float64}, s::SparseMatrixCSC{Float64})
    m, n = size(d)
    if size(s) ≠ (m, n)
        throw(DimensionMismatch("size(d) ≠ size(s)"))
    end
    if nnz(d) == nnz(s)  # FIXME: should also check that colptr members match
        copy!(nonzeros(d), nonzeros(s))
        return d
    end
    drv, srv, dnz, snz = rowvals(d), rowvals(s), nonzeros(d), nonzeros(s)
    fill!(dnz, 0)
    for j in 1:n
        dnzr = nzrange(d, j)
        dnzrv = Compat.view(drv, dnzr)
        snzr = nzrange(s, j)
        if length(snzr) == length(dnzr) && all(dnzrv .== Compat.view(srv, snzr))
            copy!(Compat.view(dnz, dnzr),Compat.view(snz, snzr))
        else
            for k in snzr
                ssr = srv[k]
                kk = searchsortedfirst(dnzrv, ssr)
                if kk > length(dnzrv) || dnzrv[kk] != ssr
                    throw(ArgumentError("cannot inject sparse s into sparse d"))
                end
                dnz[dnzr[kk]] = snz[k]
            end
        end
    end
    d
end
