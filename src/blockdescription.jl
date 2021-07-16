"""
    BlockDescription

Description of blocks of `A` and `L` in a [`LinearMixedModel`](@ref)

## Fields
* `blknms`: Vector{String} of block names
* `blkrows`: Vector{Int} of the number of rows in each block
* `ALtypes`: Matrix{String} of datatypes for blocks in `A` and `L`.

When a block in `L` is the same type as the corresponding block in `A`, it is
described with a single name, such as `Dense`.  When the types differ the entry
in `ALtypes` is of the form `Diag/Dense`, as determined by a `shorttype` method.
"""
struct BlockDescription
    blknms::Vector{String}
    blkrows::Vector{Int}
    ALtypes::Matrix{String}
end
function BlockDescription(m::LinearMixedModel)
    A = m.A
    L = m.L
    blknms = push!(string.([fnames(m)...]), "fixed")
    k = length(blknms)
    ALtypes = fill("", k, k)
    Ltypes = fill(Nothing, k, k)
    for i in 1:k, j in 1:i
        ALtypes[i, j] = shorttype(A[block(i, j)], L[block(i, j)])
    end
    return BlockDescription(blknms, [size(A[kp1choose2(i)], 1) for i in 1:k], ALtypes)
end

BlockDescription(m::GeneralizedLinearMixedModel) = BlockDescription(m.LMM)

shorttype(::UniformBlockDiagonal, ::UniformBlockDiagonal) = "BlkDiag"
shorttype(::UniformBlockDiagonal, ::Matrix) = "BlkDiag/Dense"
shorttype(::SparseMatrixCSC, ::BlockedSparse) = "Sparse"
shorttype(::Diagonal, ::Diagonal) = "Diagonal"
shorttype(::Diagonal, ::Matrix) = "Diag/Dense"
shorttype(::Matrix, ::Matrix) = "Dense"
shorttype(::SparseMatrixCSC, ::SparseMatrixCSC) = "Sparse"
shorttype(::SparseMatrixCSC, ::Matrix) = "Sparse/Dense"

function Base.show(io::IO, ::MIME"text/plain", b::BlockDescription)
    rowwidth = max(maximum(ndigits, b.blkrows) + 1, 5)
    colwidth = max(maximum(textwidth, b.blknms) + 1, 14)
    print(io, rpad("rows:", rowwidth))
    println(io, cpad.(b.blknms, colwidth)...)
    for (i, r) in enumerate(b.blkrows)
        print(io, lpad(string(r, ':'), rowwidth))
        for j in 1:i
            print(io, cpad(b.ALtypes[i, j], colwidth))
        end
        println(io)
    end
end

Base.show(io::IO, b::BlockDescription) = show(io, MIME"text/plain"(), b)
