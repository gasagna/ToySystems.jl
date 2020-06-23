using BenchmarkTools
using LinearAlgebra
using ToySystems
using Test

# //////////////////
# Utility functions
# //////////////////

# Approximate J(u)*v as (F(u+ϵv)-F(u-ϵv))/2ϵ
function FDJacobian(F, u::AbstractVector, v::AbstractVector, ϵ::Real=1e-6)
    fp = F(0.0, u .+ ϵ.*v, similar(u))
    fm = F(0.0, u .- ϵ.*v, similar(u))
    return 0.5.*(fp .- fm)./ϵ
end

@testset "constructor & call                     " begin

    F1 = ToySystems.NineModeSystemEq.NineModeSystem(100)
    F2 = ToySystems.AeroOscillatorEq.AeroOscillator(11.2)
    F3 = ToySystems.RosslerEq.Rossler((0.492, 2, 4))
    F4 = ToySystems.RosslerEq.Rossler(0.3)
    F5 = ToySystems.LorenzEq.Lorenz()
    F6 = ToySystems.KSEq.KS(10, 10, 1)
    F7 = ToySystems.Sprott94.Sprott94F()

    for (u, dudt, F) in [(zeros(9), zeros(9), F1),
                         (zeros(4), zeros(4), F2),
                         (zeros(3), zeros(3), F3),
                         (zeros(3), zeros(3), F4),
                         (zeros(3), zeros(3), F5),
                         (zeros(8), zeros(8), F6),
                         (zeros(3), zeros(3), F7)]
        @test_nowarn F(0.0, u, dudt)
        alloc = @allocated F(0.0, u, dudt)
        @test alloc == 0
    end
end

@testset "linearised equations                   " begin

    # constructor and call interface
    F1 = ToySystems.NineModeSystemEq.NineModeSystem(100)
    L1 = ToySystems.NineModeSystemEq.NineModeSystemLin(100, false)
    A1 = ToySystems.NineModeSystemEq.NineModeSystemLin(100, true)

    F2 = ToySystems.AeroOscillatorEq.AeroOscillator(11.2)
    L2 = ToySystems.AeroOscillatorEq.AeroOscillatorLin(11.2, false)
    A2 = ToySystems.AeroOscillatorEq.AeroOscillatorLin(11.2, true)

    F3 = ToySystems.RosslerEq.Rossler((0.492, 2, 4))
    L3 = ToySystems.RosslerEq.RosslerLin((0.492, 2, 4), false)
    A3 = ToySystems.RosslerEq.RosslerLin((0.492, 2, 4), true)

    F4 = ToySystems.RosslerEq.Rossler(0.2)
    L4 = ToySystems.RosslerEq.RosslerLin(0.2, false)
    A4 = ToySystems.RosslerEq.RosslerLin(0.2, true)

    F5 = ToySystems.LorenzEq.Lorenz()
    L5 = ToySystems.LorenzEq.LorenzLin(false)
    A5 = ToySystems.LorenzEq.LorenzLin(true)

    F7 = ToySystems.Sprott94.Sprott94F()
    L7 = ToySystems.Sprott94.Sprott94FLin(false)
    A7 = ToySystems.Sprott94.Sprott94FLin(true)

    for (u, dudt, v, dvdt, F, L, A) in [
            (rand(9), rand(9), rand(9), rand(9), F1, L1, A1),
            (rand(4), rand(4), rand(4), rand(4), F2, L2, A2),
            (rand(3), rand(3), rand(3), rand(3), F3, L3, A3),
            (rand(3), rand(3), rand(3), rand(3), F4, L4, A4),
            (rand(3), rand(3), rand(3), rand(3), F5, L5, A5),
            (rand(3), rand(3), rand(3), rand(3), F7, L7, A7)]
        # check call interface
        @test_nowarn L(0.0, u, dudt, v, dvdt)
        alloc = @allocated L(0.0, u, dudt, v, dvdt)
        @test alloc == 0

        @test_nowarn L(0.0, u, v, dvdt)
        alloc = @allocated L(0.0, u, v, dvdt)
        @test alloc == 0
        
        # check correctness
        a = FDJacobian(F, u, v, 1e-6)
        b = L(0.0, u, v, dvdt)
        @test maximum(abs, a .- b) < 1e-8

        # check adjoint
        a = dot(L(0.0, u, v, dvdt), u)
        b = dot(v, A(0.0, u, u, dvdt))
        @test a ≈ b
    end
end