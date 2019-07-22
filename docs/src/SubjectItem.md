````julia
julia> using DataFrames, MixedModels, RData

julia> const dat = Dict(Symbol(k)=>v for (k,v) in 
    load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")));

````



````julia
julia> mm1 = fit(LinearMixedModel, @formula(Y ~ 1+S+T+U+V+W+X+Z+(1+S+T+U+V+W+X+Z|G)+(1+S+T+U+V+W+X+Z|H)), dat[:kb07])
Error: MethodError: Cannot `convert` an object of type Tuple{Array{Float64,2},MixedModels.ReMat{Float64,8},MixedModels.ReMat{Float64,8}} to an object of type Array{Float64,2}
Closest candidates are:
  convert(::Type{Array{S,N}}, !Matched::PooledArrays.PooledArray{T,R,N,RA} where RA) where {S, T, R, N} at /home/bates/.julia/packages/PooledArrays/ufJSl/src/PooledArrays.jl:288
  convert(::Type{Array{T,N}}, !Matched::StaticArrays.SizedArray{S,T,N,M} where M) where {T, S, N} at /home/bates/.julia/packages/StaticArrays/3KEjZ/src/SizedArray.jl:62
  convert(::Type{T<:Array}, !Matched::AbstractArray) where T<:Array at array.jl:474
  ...

````



````julia
julia> mm1.optsum
Error: UndefVarError: mm1 not defined

````



````julia
julia> mm1.trms[1].Λ
Error: UndefVarError: mm1 not defined

````



````julia
julia> mm1.trms[2].Λ
Error: UndefVarError: mm1 not defined

````


