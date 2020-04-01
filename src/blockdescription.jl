"""
    BlockDescription

Description of blocks of `A` and `L` in a [`LinearMixedModel`](@ref)

## Fields
* `blknms`: Vector of block names as `String`s
* `blkrows`: Vector of number of rows in each block
* `Atypes`: Matrix of `DataType`s for blocks in `A`
* `Ltypes`: Matrix of `DataType`s of blocks in `L`
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
        ALtypes[i, j] = shorttype(A[Block(i,j)], L[Block(i,j)])
    end
    BlockDescription(
        blknms,
        [size(A[Block(i, 1)], 1) for i in 1:k],
        ALtypes,
     )
end

shorttype(::UniformBlockDiagonal,::UniformBlockDiagonal) = "BlkDiag"
shorttype(::UniformBlockDiagonal,::Matrix) = "BlkDiag/Dense"
shorttype(::SparseMatrixCSC,::BlockedSparse) = "Sparse"
shorttype(::Diagonal,::Diagonal) = "Diagonal"
shorttype(::Diagonal,::Matrix) = "Diag/Dense"
shorttype(::Matrix,::Matrix) = "Dense"
shorttype(::SparseMatrixCSC,::SparseMatrixCSC) = "Sparse"

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
        println()
    end
end

@deprecate describeblocks(io, m) show(io, MIME"text/plain", BlockDescription(m))
@deprecate describeblocks(m) show(stdout, MIME"text/plain", BlockDescription(m))
