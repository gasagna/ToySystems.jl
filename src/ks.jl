module KSEq

import Flows
import SparseArrays
import SuiteSparse
import LinearAlgebra

using ToySystems: no_forcing

export KS, KSTan, energy, dfdc_forcing, denergydu_dot_v

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
        dudt[1]   = - (u[1] + eq.c)*( u[2] - 0.0   )/(2h)
        dudt[M]   = - (u[M] + eq.c)*( 0.0  - u[M-1])/(2h)
        @simd for i = 2:M-1
            dudt[i] = - (u[i] + eq.c)*(u[i+1] - u[i-1])/(2h)
        end
    end
    return dudt
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# TANGENT EQUATIONS

struct KSTan{F}
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

    # the additional forcing
    forcing::F

    function KSTan(N::Int, L::Real, c::Real, forcing=no_forcing)
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

        return new{typeof(forcing)}(N, L, c, A, map, forcing)
    end
end

# explicit bit
function (eq::KSTan{F})(t::Real,
                        u::AbstractVector,
                     dudt::AbstractVector,
                        v::AbstractVector,
                     dvdt::AbstractVector) where {F}
    # grid size
    N = eq.N
    
    # number of degrees of freedom
    M = N - 2
    
    # grid spacing
    h = eq.L/(N-1)

    # convective velocity
    c = eq.c

    # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
    # g stands for ghost point
    # g 0                 0 g
    # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅

    # the linearised operator is
    # - (u + c) * dvdx - v *dudx
    @inbounds begin
        dvdt[1] = - (u[1] + c)*( v[2] - 0.0   )/(2h) - v[1]*( u[2] - 0.0   )/(2h)
        dvdt[M] = - (u[M] + c)*( 0.0  - v[M-1])/(2h) - v[M]*( 0.0  - u[M-2])/(2h)
        @simd for i = 2:M-1
            dvdt[i] = - (u[i] + c)*(v[i+1] - v[i-1])/(2h) - v[i]*(u[i+1] - u[i-1])/(2h)
        end
    end
    
    # add forcing if needed
    eq.forcing(t, u, dudt, v, dvdt)

    return dvdt
end

function (eq::KSTan{F})(t::Real,
                        u::AbstractVector,
                        v::AbstractVector,
                     dvdt::AbstractVector) where {F} 
    return eq(t, u, u, v, dvdt)
end


# LINEAR PART
# obey Flows interface
LinearAlgebra.mul!(dudt, eq::Union{KS, KSTan}, u) = 
    LinearAlgebra.mul!(dudt, eq.A, u)

# solve z in the system (I - c*A)*z = y
# this allocates because of UMFPACK solve routines requires it
function Flows.ImcA!(eq::Union{KS, KSTan}, c::Real, y, z)
    # convert c to a float to check it's in the dict
    cf = Float64(c)

    # check we have a factorisation of I - c*A, or make one
    if cf ∉ keys(eq.map)
        eq.map[cf] = LinearAlgebra.lu(LinearAlgebra.I - cf*eq.A)
    end

    # then return
    return LinearAlgebra.ldiv!(z, eq.map[cf], y)
end


# the energy density, the integral from 0 to L of u squared in dx
function energy(u::AbstractVector, L::Real)
    # number of dofs
    M = length(u)
        
    # grid spacing
    h = L/(M+1)

    # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
    # g stands for ghost point
    # g 0                 0 g
    # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅
    #   1 2 2 2 2 2 2 2 2 1  # time counted in trapezoidal rule
    # then divide by two for the area of a trapezium
    @inbounds begin
        I = 0.5*(u[1]^2 + u[M]^2) * h
        @simd for i = 2:M-1
            I += u[i]^2 * h
        end
    end
    return I/L
end

# the dot product between the energy gradient and v
function denergydu_dot_v(u::AbstractVector, v::AbstractVector, L::Real)
    # number of dofs
    M = length(u)
        
    # grid spacing
    h = L/(M+1)

    # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
    # g stands for ghost point
    # g 0                 0 g
    # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅
    #   1 2 2 2 2 2 2 2 2 1  # time counted in trapezoidal rule
    # then divide by two for the area of a trapezium
    @inbounds begin
        I = 0.5*(u[1]*v[1] + u[M]*v[M]) * h/L
        @simd for i = 2:M-1
            I += u[i]*v[i] * h/L
        end
    end
    return 2*I
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# FORCING FUNCTIONS FOR THE LINEARISED EQUATIONS

# sensitivity with respect to ρ
# dfdc_forcing(t, u, dudt, v, dvdt) = (@inbounds dvdt[2] += u[1]; dvdt)
# dfdρ_forcing(dvdt, u) = (@inbounds dvdt .= 0; dvdt[2] += u[1]; dvdt)

struct dfdc_forcing
    L::Float64
end

function (f::dfdc_forcing)(t, u, dudt, v, dvdt)
    # number of degrees of freedom
    M = length(u)
    
    # grid spacing
    h = f.L/(M+1)

    # eg for N = 10, L = 9 (there are only M=8 degrees of freedom)
    # g stands for ghost point
    # g 0                 0 g
    # ⋅ 0 1 2 3 4 5 6 7 8 9 ⋅
    @inbounds begin
        dvdt[1]   += - ( u[2] - 0.0   )/(2h)
        dvdt[M]   += - ( 0.0  - u[M-1])/(2h)
        @simd for i = 2:M-1
            dvdt[i] += - (u[i+1] - u[i-1])/(2h)
        end
    end
    return dudt
end

end