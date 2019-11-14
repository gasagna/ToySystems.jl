module NineModeSystemEq

using LinearAlgebra
using ToySystems: no_forcing, _mayswap

export NineModeSystem,
       NineModeSystemLin

# Domain size
const Lx = 4π
const Lz = 2π

# Define system constants
const cα = 2π/Lx
const cβ = π/2
const cγ = 2π/Lz
const cβcβ = cβ*cβ
const cβcγ = cβ*cγ
const cαcα = cα*cα
const cγcγ = cγ*cγ
const cαcβcγ = cα*cβcγ
const sqrt23 = sqrt(2/3)
const sqrt32 = sqrt(3/2)
const sqrt6 = sqrt(6)
const sqrtcβcβpluscγcγ = sqrt(cβcβ + cγcγ)
const sqrtcαcαpluscγcγ = sqrt(cαcα + cγcγ)
const sqrtcαcαpluscβcβpluscγcγ = sqrt(cαcα + cβcβ + cγcγ)

# ///////////////////
# Nonlinear equations
# ///////////////////
struct NineModeSystem
    invRe::Float64
    NineModeSystem(Re::Real) = new(1/Re)
end

function (eq::NineModeSystem)(t::Real, u::AbstractVector, dudt::AbstractVector)
    invRe = eq.invRe
    @inbounds begin
        dudt[1] = cβcβ*invRe - (cβcβ*u[1])*invRe + (sqrt32*cβcγ*u[2]*u[3])/sqrtcβcβpluscγcγ- (sqrt32*cβcγ*u[6]*u[8])/sqrtcαcαpluscβcβpluscγcγ
        dudt[2] = -((((4*cβcβ)/3 + cγcγ)*u[2])*invRe) + (5*sqrt23*cγcγ*u[4]*u[6])/(3*sqrtcαcαpluscγcγ) - (cγcγ*u[5]*u[7])/(sqrt6*sqrtcαcαpluscγcγ) - (cαcβcγ*u[5]*u[8])/(sqrt6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - (sqrt32*cβcγ*(u[1]*u[3] + u[3]*u[9]))/sqrtcβcβpluscγcγ
        dudt[3] = -(((cβcβ + cγcγ)*u[3])*invRe) + (sqrt23*cαcβcγ*(u[5]*u[6] + u[4]*u[7]))/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) + ((-3*cγcγ*(cαcα + cγcγ) + cβcβ*(3*cαcα + cγcγ))*u[4]*u[8])/(sqrt6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        dudt[4] = -((3*cαcα + 4*cβcβ)*u[4])/(3/invRe) - (cα*u[1]*u[5])/sqrt6 - (5*sqrt23*cαcα*u[2]*u[6])/(3*sqrtcαcαpluscγcγ) - (sqrt32*cαcβcγ*u[3]*u[7])/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) - (sqrt32*cαcα*cβcβ*u[3]*u[8])/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - (cα*u[5]*u[9])/sqrt6
        dudt[5] = (cα*u[1]*u[4])/sqrt6 - ((cαcα + cβcβ)*u[5])*invRe + (sqrt23*cαcβcγ*u[3]*u[6])/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) +    (cαcα*u[2]*u[7])/(sqrt6*sqrtcαcαpluscγcγ) - (cαcβcγ*u[2]*u[8])/(sqrt6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) + (cα*u[4]*u[9])/sqrt6
        dudt[6] = (5*sqrt23*(cαcα - cγcγ)*u[2]*u[4])/(3*sqrtcαcαpluscγcγ) - (2*sqrt23*cαcβcγ*u[3]*u[5])/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) - ((3*cαcα + 4*cβcβ + 3*cγcγ)*u[6])/(3/invRe) + (cα*u[1]*u[7])/sqrt6 + (sqrt32*cβcγ*u[1]*u[8])/sqrtcαcαpluscβcβpluscγcγ + (cα*u[7]*u[9])/sqrt6 + (sqrt32*cβcγ*u[8]*u[9])/sqrtcαcαpluscβcβpluscγcγ
        dudt[7] = (cαcβcγ*u[3]*u[4])/(sqrt6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) + ((-cαcα + cγcγ)*u[2]*u[5])/(sqrt6*sqrtcαcαpluscγcγ) - (cα*u[1]*u[6])/sqrt6 - ((cαcα + cβcβ + cγcγ)*u[7])*invRe - (cα*u[6]*u[9])/sqrt6
        dudt[8] = (cγcγ*(3*cαcα - cβcβ + 3*cγcγ)*u[3]*u[4])/(sqrt6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) + (sqrt23*cαcβcγ*u[2]*u[5])/(sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - ((cαcα + cβcβ + cγcγ)*u[8])*invRe
        dudt[9] = (sqrt32*cβcγ*u[2]*u[3])/sqrtcβcβpluscγcγ - (sqrt32*cβcγ*u[6]*u[8])/sqrtcαcαpluscβcβpluscγcγ - (9*cβcβ*u[9])*invRe
    end
    return dudt
end


# ////////////////////
# Linearised equations
# ////////////////////
struct NineModeSystemLin{ISADJOINT, N, T<:NTuple{N, Base.Callable}}
           J::Matrix{Float64} # use this matrix for calculating the linearised operator
       invRe::Float64         # inverse of Reynolds number
    forcings::T               # a tuple of functions with signature (t, u, dudt, v, dvdt)
end

# slurp arguments
NineModeSystemLin(Re::Real, isadjoint::Bool, forcings::Vararg{Any, N}) where {N} =
    NineModeSystemLin{isadjoint, N, typeof(forcings)}(zeros(9, 9), 1/Re, forcings)

# defaults to homogeneous problem
NineModeSystemLin(Re::Real) = NineModeSystemLin(Re, no_forcing)

# Linearised equations
@generated function (eq::NineModeSystemLin{ISADJOINT, N})(t::Real,
                                                          u::AbstractVector,
                                                       dudt::AbstractVector,
                                                          v::AbstractVector,
                                                       dvdt::AbstractVector) where {ISADJOINT, N}
    quote
        # compute linear part
        _NineModeSystemJacobian(t, u, eq.J, eq.invRe, $(Val(ISADJOINT)))
        LinearAlgebra.mul!(dvdt, eq.J, v)

        # add forcing (can be nothing too)
        Base.Cartesian.@nexprs $N i->eq.forcings[i](t, u, dudt, v, dvdt)

        return dvdt
    end
end

(eq::NineModeSystemLin)(t::Real,
                        u::AbstractVector,
                        v::AbstractVector,
                     dvdt::AbstractVector) = eq(t, u, u, v, dvdt)

function _NineModeSystemJacobian(t::Real, u::AbstractVector, J::Matrix, invRe::Real, ISADJOINT)
    @inbounds begin
        u1, u2, u3, u4, u5, u6, u7, u8, u9 = u

        J[_mayswap(1, 1, ISADJOINT)...] = -cβcβ*invRe
        J[_mayswap(2, 1, ISADJOINT)...] = -sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[_mayswap(3, 1, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(4, 1, ISADJOINT)...] = -sqrt6*cα*u5/6
        J[_mayswap(5, 1, ISADJOINT)...] = sqrt6*cα*u4/6
        J[_mayswap(6, 1, ISADJOINT)...] = sqrt6*cα*u7/6 + sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[_mayswap(7, 1, ISADJOINT)...] = -sqrt6*cα*u6/6
        J[_mayswap(8, 1, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(9, 1, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(1, 2, ISADJOINT)...] = sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[_mayswap(2, 2, ISADJOINT)...] = -(4*cβcβ/3 + cγcγ)*invRe
        J[_mayswap(3, 2, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(4, 2, ISADJOINT)...] = -5/3*sqrt23*cαcα*u6/sqrtcαcαpluscγcγ
        J[_mayswap(5, 2, ISADJOINT)...] = sqrt6*cαcα*u7/(6*sqrtcαcαpluscγcγ) - sqrt6*cαcβcγ*u8/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(6, 2, ISADJOINT)...] = 5/3*sqrt23*(cαcα-cγcγ)*u4/sqrtcαcαpluscγcγ
        J[_mayswap(7, 2, ISADJOINT)...] = sqrt6*u5*(-cαcα + cγcγ)/(6*sqrtcαcαpluscγcγ)
        J[_mayswap(8, 2, ISADJOINT)...] = sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(9, 2, ISADJOINT)...] = sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ

        J[_mayswap(1, 3, ISADJOINT)...] = sqrt32*cβcγ*u2/sqrtcβcβpluscγcγ
        J[_mayswap(2, 3, ISADJOINT)...] = -sqrt32*cβcγ*(u1 + u9)/sqrtcβcβpluscγcγ
        J[_mayswap(3, 3, ISADJOINT)...] = -(cβcβ + cγcγ)*invRe
        J[_mayswap(4, 3, ISADJOINT)...] = -sqrt32*cαcα*cβcβ*u8/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - sqrt32*cαcβcγ*u7/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(5, 3, ISADJOINT)...] = sqrt23*cαcβcγ*u6/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(6, 3, ISADJOINT)...] = -2*sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(7, 3, ISADJOINT)...] = sqrt6*cαcβcγ*u4/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(8, 3, ISADJOINT)...] = sqrt6*cγcγ*u4*(3*cαcα - cβcβ + 3*cγcγ)/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(9, 3, ISADJOINT)...] = sqrt32*cβcγ*u2/sqrtcβcβpluscγcγ

        J[_mayswap(1, 4, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 4, ISADJOINT)...] = 5/3*sqrt23*cγcγ*u6/sqrtcαcαpluscγcγ
        J[_mayswap(3, 4, ISADJOINT)...] = sqrt23*cαcβcγ*u7/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) + (cβcβ*(3*cαcα+cγcγ)-3*cγcγ*(cαcα+cγcγ))*u8/(sqrt6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(4, 4, ISADJOINT)...] = -(3*cαcα + 4*cβcβ)/(3/invRe)
        J[_mayswap(5, 4, ISADJOINT)...] = sqrt6*cα*u1/6 + sqrt6*cα*u9/6
        J[_mayswap(6, 4, ISADJOINT)...] = 5/3*sqrt23*u2*(cαcα-cγcγ)/sqrtcαcαpluscγcγ
        J[_mayswap(7, 4, ISADJOINT)...] = sqrt6*cαcβcγ*u3/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(8, 4, ISADJOINT)...] = sqrt6*cγcγ*u3*(3*cαcα - cβcβ + 3*cγcγ)/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(9, 4, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(1, 5, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 5, ISADJOINT)...] = -sqrt6*cαcβcγ*u8/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - sqrt6*cγcγ*u7/(6*sqrtcαcαpluscγcγ)
        J[_mayswap(3, 5, ISADJOINT)...] = sqrt23*cαcβcγ*u6/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(4, 5, ISADJOINT)...] = -sqrt6*cα*u1/6 - sqrt6*cα*u9/6
        J[_mayswap(5, 5, ISADJOINT)...] = -(cαcα + cβcβ)*invRe
        J[_mayswap(6, 5, ISADJOINT)...] = -2*sqrt23*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(7, 5, ISADJOINT)...] = sqrt6*u2*(-cαcα + cγcγ)/(6*sqrtcαcαpluscγcγ)
        J[_mayswap(8, 5, ISADJOINT)...] = sqrt23*cαcβcγ*u2/(sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(9, 5, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(1, 6, ISADJOINT)...] = -sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[_mayswap(2, 6, ISADJOINT)...] = 5/3*sqrt23*cγcγ*u4/sqrtcαcαpluscγcγ
        J[_mayswap(3, 6, ISADJOINT)...] = sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(4, 6, ISADJOINT)...] = -5/3*sqrt23*cαcα*u2/sqrtcαcαpluscγcγ
        J[_mayswap(5, 6, ISADJOINT)...] = sqrt23*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(6, 6, ISADJOINT)...] = -(3*cαcα + 4*cβcβ + 3*cγcγ)/(3/invRe)
        J[_mayswap(7, 6, ISADJOINT)...] = -sqrt6*cα*u1/6 - sqrt6*cα*u9/6
        J[_mayswap(8, 6, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(9, 6, ISADJOINT)...] = -sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ

        J[_mayswap(1, 7, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 7, ISADJOINT)...] = -sqrt6*cγcγ*u5/(6*sqrtcαcαpluscγcγ)
        J[_mayswap(3, 7, ISADJOINT)...] = sqrt23*cαcβcγ*u4/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(4, 7, ISADJOINT)...] = -sqrt32*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[_mayswap(5, 7, ISADJOINT)...] = sqrt6*cαcα*u2/(6*sqrtcαcαpluscγcγ)
        J[_mayswap(6, 7, ISADJOINT)...] = sqrt6*cα*u1/6 + sqrt6*cα*u9/6
        J[_mayswap(7, 7, ISADJOINT)...] = -(cαcα + cβcβ + cγcγ)*invRe
        J[_mayswap(8, 7, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(9, 7, ISADJOINT)...] = zero(eltype(J))

        J[_mayswap(1, 8, ISADJOINT)...] = -sqrt32*cβcγ*u6/sqrtcαcαpluscβcβpluscγcγ
        J[_mayswap(2, 8, ISADJOINT)...] = -sqrt6*cαcβcγ*u5/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(3, 8, ISADJOINT)...] = sqrt6*u4*(cβcβ*(3*cαcα + cγcγ) - 3*cγcγ*(cαcα + cγcγ))/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(4, 8, ISADJOINT)...] = -sqrt32*cαcα*cβcβ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(5, 8, ISADJOINT)...] = -sqrt6*cαcβcγ*u2/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[_mayswap(6, 8, ISADJOINT)...] = sqrt32*cβcγ*u1/sqrtcαcαpluscβcβpluscγcγ + sqrt32*cβcγ*u9/sqrtcαcαpluscβcβpluscγcγ
        J[_mayswap(7, 8, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(8, 8, ISADJOINT)...] = -(cαcα + cβcβ + cγcγ)*invRe
        J[_mayswap(9, 8, ISADJOINT)...] = -sqrt32*cβcγ*u6/sqrtcαcαpluscβcβpluscγcγ

        J[_mayswap(1, 9, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(2, 9, ISADJOINT)...] = -sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[_mayswap(3, 9, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(4, 9, ISADJOINT)...] = -sqrt6*cα*u5/6
        J[_mayswap(5, 9, ISADJOINT)...] = sqrt6*cα*u4/6
        J[_mayswap(6, 9, ISADJOINT)...] = sqrt6*cα*u7/6 + sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[_mayswap(7, 9, ISADJOINT)...] = -sqrt6*cα*u6/6
        J[_mayswap(8, 9, ISADJOINT)...] = zero(eltype(J))
        J[_mayswap(9, 9, ISADJOINT)...] = -9*cβcβ*invRe
    end
    return J
end

end