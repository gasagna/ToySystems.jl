module KSEq

import Flows
import SparseArrays
import SuiteSparse
import LinearAlgebra

export KS, energy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# EQUATIONS

struct KS
    # number of points, including boundaries. There are N-2 degrees of freedom
    N::Int
    
    # the domain from zero to L
    L::Float64

    # the advection velocity
    c::Float64

    # linear operator
    A::SparseArrays.SparseMatrixCSC{Float64, Int64}

    # objects to solve linear problems quickly
    map::Dict{Float64, SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}
    
    function KS(N::Int, L::Real, c::Real)
        # grid spacing
        h = L/(N-1)

        # size of the state vector
        M = N - 2

        # discretised linear operators
        D2 = SparseArrays.spdiagm( 0=>fill(-2/h^2, M),
                                   1=>fill( 1/h^2, M-1),
                                  -1=>fill( 1/h^2, M-1))

        # general 
        D4 = SparseArrays.spdiagm( 0 => fill( 6/h^4, M),
                                   1 => fill(-4/h^4, M-1),
                                  -1 => fill(-4/h^4, M-1),
                                   2 => fill( 1/h^4, M-2),
                                  -2 => fill( 1/h^4, M-2))
        # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
        # g stands for ghost point
        # g 0                 0 g
        # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅
        # the neumann BC imply
        #  u[1] = u[-1]
        #  u[8] = u[10]
        D4[  1,   1] += 1/h^4
        D4[end, end] += 1/h^4

        A = - (D2 + D4)

        # dictionary with the precomputed factorisations
        map = Dict{Float64, SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}()

        # create object
        new(N, L, c, A, map)
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
    # grid size
    N = eq.N
    
    # number of degrees of freedom
    M = N - 2
    
    # grid spacing
    h = eq.L/(N-1)

    # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
    # g stands for ghost point
    # g 0                 0 g
    # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅
    @inbounds begin
        dudt[1]   = - u[1]*( u[2] - 0.0   )/(2h)
        dudt[M]   = - u[M]*( 0.0  - u[M-1])/(2h)
        @simd for i = 2:M-1
            dudt[i] = - (u[i] + eq.c)*(u[i+1] - u[i-1])/(2h)
        end
    end
    return dudt
end

# the energy density, the integral from 0 to L of u squared in dx
function energy(u::AbstractVector, L::Real)
    M = length(u)
    @inbounds begin
        I = u[1]^2
        @simd for i = 2:M
            I += u[i]^2
        end
    end
    return I/2L
end

end