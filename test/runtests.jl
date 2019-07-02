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

    for (u, dudt, F) in [(zeros(9), zeros(9), F1), ]
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

    for (u, dudt, v, dvdt, F, L, A) in [(rand(9), rand(9), rand(9), rand(9), F1, L1, A1), ]
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