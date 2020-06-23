# Details of the gradient evaluation for LMMs

Two of the fields in a `LinearMixedModel` are `BlockArray`s, `A` and `L`, plus a vector of `BlockArray`s named `Ldot` and a `Vector{NTuple{3,Int}}` named `parmap` that maps elements of the parameter vector `θ` to positions in the `λ` fields of the `reterms` field.

The Λ matrix is never explicitly created.  Multiplications by Λ' or Λ are performed blockwise and in-place by methods for `lmulΛ!` and `rmulΛ!`.

To update `L` and `Ldot` the steps are:

1. Copy the lower-triangular blocks of `A` to `L` and zero the lower-triangular blocks of each element of `Ldot`.
    a. One caveat here, each element of `Ldot` will end up with exactly one nonzero diagonal block.  Blocks above that are not touched and do not need re-zeroing.
2. Form the blocks of `Λ'A` in the corresponding blocks of `L`
3. For each element of `Ldot` use copy operations to fill its blocks with `Λ'AΛ̇  + Λ̇'AΛ`
    a. The arrays `Λ̇ `are never formed explicitly - all the information needed to perform the multiplications are in the 3-tuples in `parmap`.
    b. If the parmap element is `(i, j, k)` then right-multiplication by `Λ̇ ` affects only the i'th block of columns of `Ldot` and consists of copying the `j`'th sub-column to the k'th subcolumn position.  See the methods for `rmulΛdot!`.
    c. When `i = 1` in the parmap, it is sufficient to create `Λ'AΛ̇ ` and symmetrize only the diagonal blocks.  When `i > 1` the blocks to the left of the `i`th diagonal block must be evaluated.  This requires the corresponding block of `Λ̇'AΛ`.  It appears that it will be necessary to evaluate the block of `AΛ` somewhere. Unfortunately, this is usually a large, dense block in a subject-item design.  Maybe allocate, say `L[Block(1,2)]`, to accumulate the product? 
