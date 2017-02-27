"""
    inject!{T <: AbstractMatrix, S <: AbstractMatrix}(d::T, s::S)

Behaves like `copy!(d, s)` allowing for heterogeneous matrix types.
"""
inject!(d,s) = copy!(d,s)               # fallback method

if false
    function inject!(d::UpperTriangular, s::UpperTriangular)
        if (n = size(s, 2)) ≠ size(d, 2)
            throw(DimensionMismatch("size(s, 2) ≠ size(d, 2)"))
        end
        for j in 1:n
            inject!(view(d, 1 : j, j), view(s, 1 : j, j))
        end
        d
    end
end

function inject!{T<:Real}(d::StridedMatrix{T}, s::Diagonal{T})
    sd = s.diag
    if length(sd) ≠ min(size(d)...)
        throw(DimensionMismatch("min(size(d)...) ≠ size(s, 2)"))
    end
    fill!(d, 0)
    @inbounds for i in eachindex(sd)
        d[i, i] = sd[i]
    end
    d
end

function inject!{T<:AbstractFloat}(d::Diagonal{T}, s::Diagonal{T})
    copy!(d.diag, s.diag)
    d
end

function inject!{T<:AbstractFloat}(d::Diagonal{LowerTriangular{T,Matrix{T}}},
    s::Diagonal{Matrix{T}})
    @assert length(d) == length(s)
    sd = s.diag
    dd = d.diag
    for k in eachindex(dd)
        copy!(dd[k].data, sd[k])
    end
    d
end

function inject!(d::SparseMatrixCSC{Float64}, s::SparseMatrixCSC{Float64})
    m, n = size(d)
    @assert size(s) == (m, n)
    if nnz(d) == nnz(s)  # FIXME: should also check that colptr members match
        copy!(nonzeros(d), nonzeros(s))
        return d
    end
    drv, srv, dnz, snz = rowvals(d), rowvals(s), nonzeros(d), nonzeros(s)
    fill!(dnz, 0)
    for j in 1:n
        dnzr = nzrange(d, j)
        dnzrv = view(drv, dnzr)
        snzr = nzrange(s, j)
        if length(snzr) == length(dnzr) && all(dnzrv .== view(srv, snzr))
            copy!(view(dnz, dnzr),view(snz, snzr))
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
