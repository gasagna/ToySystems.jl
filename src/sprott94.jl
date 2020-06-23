module Sprott94

using LinearAlgebra
using ToySystems: no_forcing, _mayswap

export Sprott94F, Sprott94FLin

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# NONLINEAR EQUATIONS

struct Sprott94F  end


function (eq::Sprott94F)(t, u, dudt)
    x, y, z = u
    @inbounds dudt[1] =           y + z
    @inbounds dudt[2] =  -x + 0.5*y
    @inbounds dudt[3] =   x^2       - z
    return dudt
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# FORCING FUNCTIONS FOR THE LINEARISED EQUATIONS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# LINEARISED EQUATIONS

struct Sprott94FLin{F, ISADJOINT}
          J::Matrix{Float64}
    forcing::F
end

Sprott94FLin(isadjoint::Bool=false, forcing::F = no_forcing) where {F} =
    Sprott94FLin{F, isadjoint}(zeros(3, 3), forcing)

@generated function (eq::Sprott94FLin{F, ISADJOINT})(t::Real,
                                                     u::AbstractVector,
                                                  dudt::AbstractVector,
                                                     v::AbstractVector,
                                                  dvdt::AbstractVector) where {F, ISADJOINT}
    quote
        _Sprott94FJacoabian(t, u, eq.J, $(Val(ISADJOINT)))
        LinearAlgebra.mul!(dvdt, eq.J, v)
        eq.forcing(t, u, dudt, v, dvdt)
        return dvdt
    end
end

(eq::Sprott94FLin)(t::Real,
                   u::AbstractVector,
                   v::AbstractVector,
                dvdt::AbstractVector) = eq(t, u, u, v, dvdt)

function _Sprott94FJacoabian(t::Real,
                             u::AbstractVector,
                             J::Matrix,
                     ISADJOINT::Val)
    x, y, z = u
    @inbounds begin
        J[_mayswap(1, 1, ISADJOINT)...] =  0.0
        J[_mayswap(1, 2, ISADJOINT)...] =  1.0
        J[_mayswap(1, 3, ISADJOINT)...] =  1.0
        J[_mayswap(2, 1, ISADJOINT)...] = -1.0
        J[_mayswap(2, 2, ISADJOINT)...] =  0.5
        J[_mayswap(2, 3, ISADJOINT)...] =  0.0
        J[_mayswap(3, 1, ISADJOINT)...] =  2*x
        J[_mayswap(3, 2, ISADJOINT)...] =  0.0
        J[_mayswap(3, 3, ISADJOINT)...] = -1.0
    end
    return J
end

end