module AeroOscillatorEq

using LinearAlgebra
using ToySystems: no_forcing, _mayswap

export AeroOscillator,
       AeroOscillatorLin,
       dfdQ_forcing


# Problem parameters
const _M =         [1.00  0.25;
                    0.25  0.50]
const _D = inv(_M)*[0.10  0.00;
                    0.00  0.10]
const _A = inv(_M)*[0.20  0.00;
                    0.00  0.50]
const _B = inv(_M)*[0.00  0.10;
                    0.00 -0.10]
const _C = inv(_M)*[0.00  0.00;
                    0.00 20.00];

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# NONLINEAR EQUATIONS

struct AeroOscillator
    Q::Float64
end

function (eq::AeroOscillator)(t::Real, x::AbstractVector, dxdt::AbstractVector)
    @assert length(x) == length(dxdt) == 4
    @inbounds begin
        dxdt[1] = x[3]
        dxdt[2] = x[4]
        dxdt[3] = (-  _D[1, 1]*x[3]   - _D[1, 2]*x[4]
                   -  _A[1, 1]*x[1]   - _A[1, 2]*x[2]
                   - (_B[1, 1]*x[1]   + _B[1, 2]*x[2])*eq.Q
                   -  _C[1, 1]*x[1]^3 - _C[1, 2]*x[2]^3)
        dxdt[4] = (-  _D[2, 1]*x[3]   - _D[2, 2]*x[4]
                   -  _A[2, 1]*x[1]   - _A[2, 2]*x[2]
                   - (_B[2, 1]*x[1]   + _B[2, 2]*x[2])*eq.Q
                   -  _C[2, 1]*x[1]^3 - _C[2, 2]*x[2]^3)
    end
    return dxdt
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# LINEARISED EQUATIONS

struct AeroOscillatorLin{F, ISADJOINT}
          J::Matrix{Float64} # use this matrix for calculating the linearised operator
          Q::Float64
    forcing::F
end

AeroOscillatorLin(Q::Real, isadjoint::Bool, forcing::F = no_forcing) where {F} =
    AeroOscillatorLin{F, isadjoint}(zeros(4, 4), Q, forcing)

@generated function (eq::AeroOscillatorLin{F, ISADJOINT})(t::Real,
                                                          u::AbstractVector,
                                                       dudt::AbstractVector,
                                                          v::AbstractVector,
                                                       dvdt::AbstractVector) where {F, ISADJOINT}
    quote
        _AeroOscillatorJacobian(t, u, eq.J, eq.Q, $(Val(ISADJOINT)))
        LinearAlgebra.mul!(dvdt, eq.J, v)
        eq.forcing(t, u, dudt, v, dvdt)
        return dvdt
    end
end

function _AeroOscillatorJacobian(t::Real,
                                 u::AbstractVector,
                                 J::Matrix,
                                 Q::Real, ISADJOINT::Val)
    
    @inbounds begin
        J[_mayswap(1, 1, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(1, 2, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(1, 3, ISADJOINT)...] =  one(eltype(J))
        J[_mayswap(1, 4, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(2, 1, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 2, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 3, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 4, ISADJOINT)...] =  one(eltype(J))

        J[_mayswap(3, 1, ISADJOINT)...] = -(_A[1, 1] + _B[1, 1]*Q + 3*_C[1, 1]*u[1]^2)
        J[_mayswap(3, 2, ISADJOINT)...] = -(_A[1, 2] + _B[1, 2]*Q + 3*_C[1, 2]*u[2]^2)
        J[_mayswap(3, 3, ISADJOINT)...] = - _D[1, 1]
        J[_mayswap(3, 4, ISADJOINT)...] = - _D[1, 2]

        J[_mayswap(4, 1, ISADJOINT)...] = -(_A[2, 1] + _B[2, 1]*Q + 3*_C[2, 1]*u[1]^2)
        J[_mayswap(4, 2, ISADJOINT)...] = -(_A[2, 2] + _B[2, 2]*Q + 3*_C[2, 2]*u[2]^2)
        J[_mayswap(4, 3, ISADJOINT)...] = - _D[2, 1]
        J[_mayswap(4, 4, ISADJOINT)...] = - _D[2, 2]
    end
    return J
end

(eq::AeroOscillatorLin)(t::Real,
                        u::AbstractVector,
                        v::AbstractVector,
                     dvdt::AbstractVector) = eq(t, u, u, v, dvdt)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# FORCING TERMS

# sensitivity ∂f/∂Q
dfdQ_forcing(t, x, dxdt, y, dydt) =
    (@inbounds dydt[3] += -(_B[1, 1]*x[1] + _B[1, 2]*x[2]);
     @inbounds dydt[4] += -(_B[2, 1]*x[1] + _B[2, 2]*x[2]); dydt)

end