module LorenzEq

using LinearAlgebra
using ToySystems: no_forcing, _mayswap

export Lorenz,
       LorenzLin

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# NONLINEAR EQUATIONS

struct Lorenz end

function (eq::Lorenz)(t, u, dudt)
    x, y, z = u
    @inbounds dudt[1] =  10 * (y - x)
    @inbounds dudt[2] =  28 *  x - y - x*z
    @inbounds dudt[3] = -8/3 * z + x*y
    return dudt
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# FORCING FUNCTIONS FOR THE LINEARISED EQUATIONS

# sensitivity with respect to rho
dfdρ_forcing(t, u, dudt, v, dvdt) = (@inbounds dvdt[2] += u[1]; dvdt)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# LINEARISED EQUATIONS

struct LorenzLin{F, ISADJOINT}
          J::Matrix{Float64}
    forcing::F
end

LorenzLin(isadjoint::Bool, forcing::F = no_forcing) where {F} =
    LorenzLin{F, isadjoint}(zeros(3, 3), forcing)

@generated function (eq::LorenzLin{F, ISADJOINT})(t::Real,
                                                  u::AbstractVector,
                                               dudt::AbstractVector,
                                                  v::AbstractVector,
                                               dvdt::AbstractVector) where {F, ISADJOINT}
    quote
        _LorenzJacobian(t, u, eq.J, $(Val(ISADJOINT)))
        LinearAlgebra.mul!(dvdt, eq.J, v)
        eq.forcing(t, u, dudt, v, dvdt)
        return dvdt
    end
end

(eq::LorenzLin)(t::Real,
                u::AbstractVector,
                v::AbstractVector,
             dvdt::AbstractVector) = eq(t, u, u, v, dvdt)

function _LorenzJacobian(t::Real,
                         u::AbstractVector,
                         J::Matrix,
                 ISADJOINT::Val)
    x, y, z = u
    @inbounds begin
        J[_mayswap(1, 1, ISADJOINT)...] = -10
        J[_mayswap(1, 2, ISADJOINT)...] =  10
        J[_mayswap(1, 3, ISADJOINT)...] =  0
        J[_mayswap(2, 1, ISADJOINT)...] =  28 - z
        J[_mayswap(2, 2, ISADJOINT)...] =  -1
        J[_mayswap(2, 3, ISADJOINT)...] =  -x
        J[_mayswap(3, 1, ISADJOINT)...] =  y
        J[_mayswap(3, 2, ISADJOINT)...] =  x
        J[_mayswap(3, 3, ISADJOINT)...] =  -8/3
    end
    return J
end

end