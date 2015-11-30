### Test error exits
mf = ModelFrame(Yield ~ 1 + (1|Batch), ds)
Λ = [LowerTriangular(eye(1)) for i = 1]
y = convert(Vector,ds[:Yield])
Rem = push!([],remat(:(1|Batch), ds))
X = ones(30,1)

@test_throws ArgumentError LinearMixedModel(mf,ones(1),Λ,X,y,Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,Rem,Λ,X,y[1:20],Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,Rem,Λ,X[1:20,:],y,Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,append!(copy(Rem),Rem),Λ,X,y,Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,Rem,append!(copy(Λ),Λ),X,y,Float64[])
@test_throws DimensionMismatch LinearMixedModel(mf,Rem,[LowerTriangular(eye(2)) for i in 1],X,y,Float64[])
@test_throws ArgumentError LinearMixedModel(mf,Rem,Λ,X,y,fill(-1.,30))

@test_throws ArgumentError lmm(Yield ~ 1,ds)

modl = lmm(Yield ~ 1 + (1|Batch), ds);

@test_throws ArgumentError modl[:ϕ] = [1.]
@test_throws DimensionMismatch modl[:θ] = [0.,1.]
