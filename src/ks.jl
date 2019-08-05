module KSEq

import Flows
import SparseArrays
import SuiteSparse
import LinearAlgebra

export KS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# EQUATIONS

struct KS
    # number of interior points, excluding boundaries
    N::Int
    # the domain from zero to L
    L::Float64
    # linear operator
    A::SparseArrays.SparseMatrixCSC{Float64, Int64}
    # objects to solve linear problems quickly
    map::Dict{Float64, SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}
    function KS(N::Int, L::Real)
        # grid size
        h = L/(N+3)

        # discretised linear operators
        D2 = SparseArrays.spdiagm( 0=>fill(-2/h^2, N),
                                   1=>fill( 1/h^2, N-1),
                                  -1=>fill( 1/h^2, N-1))

        # general 
        D4 = SparseArrays.spdiagm( 0 => fill( 6/h^4, N),
                                   1 => fill(-4/h^4, N-1),
                                  -1 => fill(-4/h^4, N-1),
                                   2 => fill( 1/h^4, N-2),
                                  -2 => fill( 1/h^4, N-2))
        # ex for N = 6
        # x x             x x
        # 0 1 2 3 4 5 6 7 8 9
        # 4u[1] - u[2] = 0 -> u[1] = 0.25*u[2]
        # 4u[8] - u[7] = 0 -> u[8] = 0.25*u[7]
        D2[  1,   1] +=  1 * 0.25/h^2
        D2[end, end] +=  1 * 0.25/h^2
        D4[  1,   1] += -4 * 0.25/h^4
        D4[end, end] += -4 * 0.25/h^4

        A = - (D2 + D4)

        # dictionary with the precomputed factorisations
        map = Dict{Float64, SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}()

        # create object
        new(N, L, A, map)
    end
end

# obey Flows interface
LinearAlgebra.mul!(dudt, eq::KS, u) = 
    LinearAlgebra.mul!(dudt, eq.A, u)

# solve z in the system (I - c*A)*z = y
# this allocates because of UMFPACK solve routines requires it
function Flows.ImcA!(eq::KS, c::Real, y, z)
    # convert c to a float to check it's in the dict
    cf = Float64(c)

    # check we have a factorisation of I - c*A, or make one
    if cf ∉ keys(eq.map)
        eq.map[cf] = LinearAlgebra.lu(LinearAlgebra.I - cf*eq.A)
    end

    # then return
    return LinearAlgebra.ldiv!(z, eq.map[cf], y)
end

# nonlinear term
function (eq::KS)(t, u, dudt)
    N = eq.N
    h = eq.L/(N+3)
    # u .-= sum(u)/N
    # ex for N = 6
    # x x             x x
    # 0 1 2 3 4 5 6 7 8 9
    # 4u[1] - u[2] = 0 -> u[1] = 0.25*u[2]
    # 4u[8] - u[7] = 0 -> u[8] = 0.25*u[7]
    @inbounds begin
        dudt[1]   = - u[1]*(     u[2] - 0.25*u[1]  )/(2h)
        dudt[N]   = - u[N]*(0.25*u[N] -      u[N-1])/(2h)
        @simd for i = 2:eq.N-1
            dudt[i] = - u[i]*(u[i+1] - u[i-1])/(2h)
        end
    end
    return dudt
end


end