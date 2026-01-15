# using FiniteDiff
using LinearAlgebra
using MixedModels
using Test
using MixedModels: Omega_dot_diag_block!

using MixedModelsDatasets: dataset

include("modelcache.jl")

@testset "gradient" begin
    @testset "single_scalar" begin
        fm1 = only(models(:dyestuff2))
        θ = fm1.θ
        blk = Omega_dot_diag_block!(similar(first(fm1.A)), fm1, 1)
        @test all(≈(10. * only(θ)), blk.diag)
        Omega_dot_diag_block!(blk, updateL!(setθ!(fm1, ones(1))), 1)
        @test all(==(10.), blk.diag)
        updateL!(setθ!(fm1, θ))         # restore the estimated parameter values

        fm2 = first(models(:pastes))
        θ = fm2.θ
        blk = Omega_dot_diag_block!(similar(first(fm2.A)), fm2, 1)
        @test all(≈(4. * only(θ)), blk.diag)
        Omega_dot_diag_block!(blk, updateL!(setθ!(fm2, ones(1))), 1)
        @test all(==(4.), blk.diag)
        updateL!(setθ!(fm2, θ))

        fm3 = last(models(:pastes))     # first of two nested, scalar r.e. terms
        θ = fm3.θ
        blk = Omega_dot_diag_block!(similar(first(fm3.A)), fm3, 1)
        @test all(≈(4. * first(θ)), blk.diag)

        fm4 = only(models(:penicillin))
        blk = Omega_dot_diag_block!(similar(first(fm4.A)), fm4, 1)
        @test all(≈(12. * first(fm4.θ)), blk.diag)

        fm5 = first(models(:sleepstudy))
        blk = Omega_dot_diag_block!(similar(first(fm5.A)), fm5, 1)
        @test all(≈(20. * only(fm5.θ)), blk.diag)
    end
    @testset "single_vector" begin
        fm6 = last(models(:sleepstudy))
        λ = only(fm6.reterms).λ
        θ = fm6.θ
        blk = Omega_dot_diag_block!(UniformBlockDiagonal(similar(first(fm6.L).data)), fm6, 1)
        blk_dat = blk.data
        A11_dat = first(fm6.A).data
        @test all(≈(20. * first(θ)), view(blk_dat, 1, 1, :))
        @test all(iszero, view(blk_dat, 2, 2, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk_dat, 2, 1, :))
        odiag = dot(view(λ, 2, :), view(A11_dat, :, 1, 1))
        @test all(≈(odiag), view(blk_dat, 1, 2, :))

        Omega_dot_diag_block!(blk, fm6, 2)
        @test all(iszero, view(blk_dat, 1, 1, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk, 2, 1, :))   # result should be symmetric
        @test all(==(10. * first(θ)), view(blk_dat, 1, 2, :))
        diag2 = 2. * dot(view(λ.data, 2, :), view(A11_dat, :, 1, 1))
        @test all(≈(diag2), view(blk_dat, 2, 2, :))

        Omega_dot_diag_block!(blk, fm6, 3)
        @test all(iszero, view(blk_dat, 1, 1, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk, 2, 1, :))   # faces of result should be symmetric
        @test all(≈(45. * first(θ)), view(blk_dat, 1, 2, :))
        diag2 = 2. * dot(view(λ.data, 2, :), view(A11_dat, :, 2, 1))
        @test all(≈(diag2), view(blk_dat, 2, 2, :))

#        FiniteDiff.finite_difference_gradient(objective!(fm6), θ)
#        ldfun(m::LinearMixedModel, θ::Vector{Float64}) = logdet(updateL!(setθ!(m, θ)))

    end
end