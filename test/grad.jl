# using FiniteDiff
using LinearAlgebra
using MixedModels
using Test
using MixedModels: grad_blocks, eval_grad_p!, initialize_blocks!

include("modelcache.jl")

@testset "gradient" begin
    @testset "single_scalar" begin
        fm1 = only(models(:dyestuff))
        θ = fm1.θ
        blks = initialize_blocks!(grad_blocks(fm1), fm1, 1)
        dblk = blks[1,1].diag
        @test all(≈(10. * only(θ)), dblk)
        ldiv_blk = ldiv!(first(fm1.L), dblk)
        @test all(≈(10. * only(θ) / first(first(fm1.L))), dblk)
        updateL!(setθ!(fm1, θ))         # restore the estimated parameter values

        fm2 = first(models(:pastes))
        θ = fm2.θ
        blks = initialize_blocks!(grad_blocks(fm2), fm2, 1)
        @test all(≈(4. * only(θ)), blks[1, 1].diag)
        initialize_blocks!(blks, updateL!(setθ!(fm2, ones(1))), 1)
        @test all(==(4.), blks[1, 1].diag)
        updateL!(setθ!(fm2, θ))

        fm3 = last(models(:pastes))     # first of two nested, scalar r.e. terms
        θ = fm3.θ
        blks = initialize_blocks!(grad_blocks(fm3), fm3, 1)

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
        blk = Omega_dot_diag_block!(similar(first(fm6.L)), fm6, 1)
        blk_dat = blk.data
        A11_dat = first(fm6.A).data
        @test all(≈(20. * first(θ)), view(blk_dat, 1, 1, :))
        @test all(iszero, view(blk_dat, 2, 2, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk_dat, 2, 1, :))
        odiag = dot(view(λ, 2, :), view(A11_dat, :, 1, 1))
        @test all(≈(odiag), view(blk_dat, 1, 2, :))
        ldiv!(LowerTriangular(first(fm6.L)), blk)

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
    end
end