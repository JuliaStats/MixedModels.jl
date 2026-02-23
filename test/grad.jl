# using FiniteDiff
using LinearAlgebra
using ForwardDiff
using MixedModels
using Test
using MixedModels: grad_blocks, eval_grad_p!, initialize_blocks!, gradient!

include("modelcache.jl")

@testset "gradient" begin
    @testset "single_scalar" begin
        fm1 = only(models(:dyestuff))
        θ = fm1.θ
        blks1 = initialize_blocks!(grad_blocks(fm1), fm1, 1, 1, 1, 1)
        dblk = blks1[1,1].diag
        @test all(≈(10. * only(θ)), dblk)
        ldiv_blk = ldiv!(first(fm1.L), dblk)
        @test all(≈(10. * only(θ) / first(first(fm1.L))), dblk)
        g1 = gradient!(similar(θ), blks1, fm1)
        @test abs(only(g1)) < 1.e-6
        updateL!(setθ!(fm1, θ))         # restore the estimated parameter values

        fm2 = first(models(:pastes))
        θ = fm2.θ
        blks2 = initialize_blocks!(grad_blocks(fm2), fm2, 1, 1, 1, 1)
        @test all(≈(4. * only(θ)), blks2[1, 1].diag)
        g2 = gradient!(similar(θ), blks2, fm2)
        @test abs(only(g2)) < 9.e-6
        initialize_blocks!(blks2, updateL!(setθ!(fm2, ones(1))), 1, 1, 1, 1)
        @test all(==(4.), blks2[1, 1].diag)
        updateL!(setθ!(fm2, θ))

        fm3 = last(models(:pastes))     # first of two nested, scalar r.e. terms
        θ = fm3.θ
        blks3 = initialize_blocks!(grad_blocks(fm3), fm3, 1, 1, 1, 1)
        @test all(≈(4. * first(θ)), blks3[1,1].diag)
        g3_fd = ForwardDiff.gradient!(similar(θ), fm3)
        @test norm(g3_fd) < 9.e-5
        g3_an = gradient!(similar(θ), blks3, fm3)

        fm4 = only(models(:penicillin))
        θ4 = fm4.θ
        blks4 = initialize_blocks!(grad_blocks(fm4), fm4, 1, 1, 1, 1)
        @test all(≈(12. * first(fm4.θ)), blks4[1,1].diag)
        g4_fd = ForwardDiff.gradient!(similar(θ4), fm4)
        g4_ad = gradient!(similar(g4_fd), blks4, fm4)     # not properly evaluated yet

        fm5 = first(models(:sleepstudy))
        θ5 = fm5.θ
        blks5 = initialize_blocks!(grad_blocks(fm5), fm5, 1, 1, 1, 1)
        @test all(≈(20. * only(θ5)), blks5[1, 1].diag)
        g5_fd = ForwardDiff.gradient!(similar(θ5), fm5)
        @test abs(only(g5_fd)) < 5.e-7
        g5_ad = gradient!(similar(θ5), blks5, fm5)
        @test abs(only(g5_ad)) < 5.e-7

    end

    @testset "single_vector" begin
        fm6 = last(models(:sleepstudy))
        λ6 = only(fm6.reterms).λ
        θ6 = fm6.θ
        blks6 = initialize_blocks!(grad_blocks(fm6), fm6, 1, 1, 1, 2)
        blk_dat = blks6[1,1].data
        A11_dat = first(fm6.A).data
        @test all(≈(20. * first(θ6)), view(blk_dat, 1, 1, :))
        @test all(iszero, view(blk_dat, 2, 2, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk_dat, 2, 1, :))
        odiag = dot(view(λ6, 2, :), view(A11_dat, :, 1, 1))
        @test all(≈(odiag), view(blk_dat, 1, 2, :))
        ldiv!(LowerTriangular(first(fm6.L)), blks6[1,1])

        initialize_blocks!(blks6, fm6, 1, 2, 2, 2)
        @test all(iszero, view(blk_dat, 1, 1, :))
        @test all(view(blk_dat, 1, 2, :) .== view(blk_dat, 2, 1, :))   # result should be symmetric
        # @test all(==(10. * first(θ)), view(blk_dat, 1, 2, :))
        # diag2 = 2. * dot(view(λ.data, 2, :), view(A11_dat, :, 1, 1))
        # @test all(≈(diag2), view(blk_dat, 2, 2, :))

        # Omega_dot_diag_block!(blk, fm6, 3)
        # @test all(iszero, view(blk_dat, 1, 1, :))
        # @test all(view(blk_dat, 1, 2, :) .== view(blk, 2, 1, :))   # faces of result should be symmetric
        # @test all(≈(45. * first(θ)), view(blk_dat, 1, 2, :))
        # diag2 = 2. * dot(view(λ.data, 2, :), view(A11_dat, :, 2, 1))
        # @test all(≈(diag2), view(blk_dat, 2, 2, :))
    end
end