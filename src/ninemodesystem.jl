module NineModeSystemEq

export NineModeSystem,
       NineModeSystemLin,
       no_forcing,
       f_forcing

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
struct NineModeSystemLin{N, T<:NTuple{N, Base.Callable}}
           J::Matrix{Float64} # use this matrix for calculating the linearised operator
       invRe::Float64         # inverse of Reynolds number
    forcings::T               # a tuple of functions with signature (t, u, dudt, v, dvdt)
end

# slurp arguments
NineModeSystemLin(Re::Real, x::Vararg{Any, N}) where {N} =
    NineModeSystemLin{N, typeof(x)}(zeros(9, 9), 1/Re, x)

# defaults to homogeneous problem
NineModeSystemLin(Re::Real) = NineModeSystemLin(Re, no_forcing)

# Linearised equations
@generated function (eq::NineModeSystemLin{N})(t::Real,
                                               u::AbstractVector,
                                            dudt::AbstractVector,
                                               v::AbstractVector,
                                            dvdt::AbstractVector) where {N}
    quote
        # compute linear part
        _NineModeSystemJacobian(t, u, eq.J, eq.invRe)
        mul!(dvdt, eq.J, v)

        # add forcing (can be nothing too)
        Base.Cartesian.@nexprs $N i->eq.forcings[i](t, u, dudt, v, dvdt)

        return dvdt
    end
end

function _NineModeSystemJacobian(t::Real, u::AbstractVector, J::Matrix, invRe::Real)
    @inbounds begin
        u1, u2, u3, u4, u5, u6, u7, u8, u9 = u

        J[1, 1] = -cβcβ*invRe
        J[2, 1] = -sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[3, 1] = zero(eltype(J))
        J[4, 1] = -sqrt6*cα*u5/6
        J[5, 1] = sqrt6*cα*u4/6
        J[6, 1] = sqrt6*cα*u7/6 + sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[7, 1] = -sqrt6*cα*u6/6
        J[8, 1] = zero(eltype(J))
        J[9, 1] = zero(eltype(J))

        J[1, 2] = sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[2, 2] = -(4*cβcβ/3 + cγcγ)*invRe
        J[3, 2] = zero(eltype(J))
        J[4, 2] = -5/3*sqrt23*cαcα*u6/sqrtcαcαpluscγcγ
        J[5, 2] = sqrt6*cαcα*u7/(6*sqrtcαcαpluscγcγ) - sqrt6*cαcβcγ*u8/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[6, 2] = 5/3*sqrt23*(cαcα-cγcγ)*u4/sqrtcαcαpluscγcγ
        J[7, 2] = sqrt6*u5*(-cαcα + cγcγ)/(6*sqrtcαcαpluscγcγ)
        J[8, 2] = sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[9, 2] = sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ

        J[1, 3] = sqrt32*cβcγ*u2/sqrtcβcβpluscγcγ
        J[2, 3] = -sqrt32*cβcγ*(u1 + u9)/sqrtcβcβpluscγcγ
        J[3, 3] = -(cβcβ + cγcγ)*invRe
        J[4, 3] = -sqrt32*cαcα*cβcβ*u8/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - sqrt32*cαcβcγ*u7/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[5, 3] = sqrt23*cαcβcγ*u6/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[6, 3] = -2*sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[7, 3] = sqrt6*cαcβcγ*u4/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[8, 3] = sqrt6*cγcγ*u4*(3*cαcα - cβcβ + 3*cγcγ)/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[9, 3] = sqrt32*cβcγ*u2/sqrtcβcβpluscγcγ

        J[1, 4] = zero(eltype(J))
        J[2, 4] = 5/3*sqrt23*cγcγ*u6/sqrtcαcαpluscγcγ
        J[3, 4] = sqrt23*cαcβcγ*u7/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ) + (cβcβ*(3*cαcα+cγcγ)-3*cγcγ*(cαcα+cγcγ))*u8/(sqrt6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[4, 4] = -(3*cαcα + 4*cβcβ)/(3/invRe)
        J[5, 4] = sqrt6*cα*u1/6 + sqrt6*cα*u9/6
        J[6, 4] = 5/3*sqrt23*u2*(cαcα-cγcγ)/sqrtcαcαpluscγcγ
        J[7, 4] = sqrt6*cαcβcγ*u3/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[8, 4] = sqrt6*cγcγ*u3*(3*cαcα - cβcβ + 3*cγcγ)/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[9, 4] = zero(eltype(J))

        J[1, 5] = zero(eltype(J))
        J[2, 5] = -sqrt6*cαcβcγ*u8/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ) - sqrt6*cγcγ*u7/(6*sqrtcαcαpluscγcγ)
        J[3, 5] = sqrt23*cαcβcγ*u6/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[4, 5] = -sqrt6*cα*u1/6 - sqrt6*cα*u9/6
        J[5, 5] = -(cαcα + cβcβ)*invRe
        J[6, 5] = -2*sqrt23*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[7, 5] = sqrt6*u2*(-cαcα + cγcγ)/(6*sqrtcαcαpluscγcγ)
        J[8, 5] = sqrt23*cαcβcγ*u2/(sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[9, 5] = zero(eltype(J))

        J[1, 6] = -sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[2, 6] = 5/3*sqrt23*cγcγ*u4/sqrtcαcαpluscγcγ
        J[3, 6] = sqrt23*cαcβcγ*u5/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[4, 6] = -5/3*sqrt23*cαcα*u2/sqrtcαcαpluscγcγ
        J[5, 6] = sqrt23*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[6, 6] = -(3*cαcα + 4*cβcβ + 3*cγcγ)/(3/invRe)
        J[7, 6] = -sqrt6*cα*u1/6 - sqrt6*cα*u9/6
        J[8, 6] = zero(eltype(J))
        J[9, 6] = -sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ

        J[1, 7] = zero(eltype(J))
        J[2, 7] = -sqrt6*cγcγ*u5/(6*sqrtcαcαpluscγcγ)
        J[3, 7] = sqrt23*cαcβcγ*u4/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[4, 7] = -sqrt32*cαcβcγ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ)
        J[5, 7] = sqrt6*cαcα*u2/(6*sqrtcαcαpluscγcγ)
        J[6, 7] = sqrt6*cα*u1/6 + sqrt6*cα*u9/6
        J[7, 7] = -(cαcα + cβcβ + cγcγ)*invRe
        J[8, 7] = zero(eltype(J))
        J[9, 7] = zero(eltype(J))

        J[1, 8] = -sqrt32*cβcγ*u6/sqrtcαcαpluscβcβpluscγcγ
        J[2, 8] = -sqrt6*cαcβcγ*u5/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[3, 8] = sqrt6*u4*(cβcβ*(3*cαcα + cγcγ) - 3*cγcγ*(cαcα + cγcγ))/(6*sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[4, 8] = -sqrt32*cαcα*cβcβ*u3/(sqrtcαcαpluscγcγ*sqrtcβcβpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[5, 8] = -sqrt6*cαcβcγ*u2/(6*sqrtcαcαpluscγcγ*sqrtcαcαpluscβcβpluscγcγ)
        J[6, 8] = sqrt32*cβcγ*u1/sqrtcαcαpluscβcβpluscγcγ + sqrt32*cβcγ*u9/sqrtcαcαpluscβcβpluscγcγ
        J[7, 8] = zero(eltype(J))
        J[8, 8] = -(cαcα + cβcβ + cγcγ)*invRe
        J[9, 8] = -sqrt32*cβcγ*u6/sqrtcαcαpluscβcβpluscγcγ

        J[1, 9] = zero(eltype(J))
        J[2, 9] = -sqrt32*cβcγ*u3/sqrtcβcβpluscγcγ
        J[3, 9] = zero(eltype(J))
        J[4, 9] = -sqrt6*cα*u5/6
        J[5, 9] = sqrt6*cα*u4/6
        J[6, 9] = sqrt6*cα*u7/6 + sqrt32*cβcγ*u8/sqrtcαcαpluscβcβpluscγcγ
        J[7, 9] = -sqrt6*cα*u6/6
        J[8, 9] = zero(eltype(J))
        J[9, 9] = -9*cβcβ*invRe
    end
    return J
end

end