module RosslerEq

using LinearAlgebra
using ToySystems: no_forcing, _mayswap

export Rossler,
       RosslerLin


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# NONLINEAR EQUATIONS

struct Rossler
    abc::NTuple{3, Float64}
end

Rossler(α::Real) = Rossler((0.2 + 0.09*α, 0.2 - 0.06*α, 5.7 - 1.18*α))

function (eq::Rossler)(t::Real, x::AbstractVector, dxdt::AbstractVector)
    @assert length(x) == length(dxdt) == 3
    @inbounds begin
        a, b, c = eq.abc
        dxdt[1] = -x[2] - x[3]
        dxdt[2] = x[1] + a*x[2]
        dxdt[3] = b + x[3]*(x[1] - c)
    end
    return dxdt
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# LINEARISED EQUATIONS

struct RosslerLin{F, ISADJOINT}
          J::Matrix{Float64}
        abc::NTuple{3, Float64}
    forcing::F
end

RosslerLin(α::Real, isadjoint::Bool, forcing::F = no_forcing) where {F} =
    RosslerLin((0.2 + 0.09*α, 0.2 - 0.06*α, 5.7 - 1.18*α), isadjoint, forcing)

RosslerLin(abc::NTuple{3, Real}, isadjoint::Bool, forcing::F = no_forcing) where {F} =
    RosslerLin{F, isadjoint}(zeros(3, 3), abc, forcing)

@generated function (eq::RosslerLin{F, ISADJOINT})(t::Real,
                                                   u::AbstractVector,
                                                dudt::AbstractVector,
                                                   v::AbstractVector,
                                                dvdt::AbstractVector) where {F, ISADJOINT}
    quote
        _RosslerJacobian(t, u, eq.J, eq.abc, $(Val(ISADJOINT)))
        LinearAlgebra.mul!(dvdt, eq.J, v)
        eq.forcing(t, u, dudt, v, dvdt)
        return dvdt
    end
end

function _RosslerJacobian(t::Real,
                          u::AbstractVector,
                          J::Matrix,
                        abc::NTuple{3, Real},
                  ISADJOINT::Val)
    @inbounds begin
        a, b, c = abc
        J[_mayswap(1, 1, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(1, 2, ISADJOINT)...] = -one(eltype(J))
        J[_mayswap(1, 3, ISADJOINT)...] = -one(eltype(J))

        J[_mayswap(2, 1, ISADJOINT)...] = one(eltype(J))
        J[_mayswap(2, 2, ISADJOINT)...] = a
        J[_mayswap(2, 3, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(3, 1, ISADJOINT)...] = u[3]
        J[_mayswap(3, 2, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(3, 3, ISADJOINT)...] = u[1] - c
    end
    return J
end

(eq::RosslerLin)(t::Real,
                 u::AbstractVector,
                 v::AbstractVector,
              dvdt::AbstractVector) = eq(t, u, u, v, dvdt)


end