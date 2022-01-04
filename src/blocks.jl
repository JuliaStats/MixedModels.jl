"""
    block(i, j)

Return the linear index of the `[i,j]` position ("block") in the row-major packed lower triangle.

Use the row-major ordering in this case because the result depends only on `i`
and `j`, not on the overall size of the array.

When `i == j` the value is the same as `kp1choose2(i)`.
"""
function block(i::Integer, j::Integer)
    0 < j ≤ i || throw(ArgumentError("[i,j] = [$i,$j] must be in the lower triangle"))
    return kchoose2(i) + j
end

"""
    kchoose2(k)

The binomial coefficient `k` choose `2` which is the number of elements
in the packed form of the strict lower triangle of a matrix.
"""
function kchoose2(k)      # will be inlined
    return (k * (k - 1)) >> 1
end

"""
    kp1choose2(k)

The binomial coefficient `k+1` choose `2` which is the number of elements
in the packed form of the lower triangle of a matrix.
"""
function kp1choose2(k)
    return (k * (k + 1)) >> 1
end

"""
    ltriindprs

A row-major order `Vector{NTuple{2,Int}}` of indices in the strict lower triangle.
"""
const ltriindprs = NTuple{2,Int}[]

function checkindprsk(k::Integer)
    kc2 = kchoose2(k)
    if length(ltriindprs) < kc2
        sizehint!(empty!(ltriindprs), kc2)
        for i in 1:k, j in 1:(i - 1)
            push!(ltriindprs, (i, j))
        end
    end
    return ltriindprs
end
