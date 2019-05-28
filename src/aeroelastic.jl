export AeroOscillator,
       AeroOscillatorLin
       no_forcing,
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

struct AeroOscillatorLin{F}
    forcing::F
          Q::Float64
end

AeroOscillatorLin(Q::Real, forcing::F = no_forcing) where {F} =
    AeroOscillatorLin{F}(forcing, Q)

function (eq::AeroOscillatorLin)(t::Real,
                                 x::AbstractVector,
                              dxdt::AbstractVector,
                                 y::AbstractVector,
                              dydt::AbstractVector)
    # linearised equations
    @inbounds begin
        dydt[1] = y[3]
        dydt[2] = y[4]
        dydt[3] = ( -(_A[1, 1] + _B[1, 1]*eq.Q + 3*_C[1, 1]*x[1]^2)*y[1]
                    -(_A[1, 2] + _B[1, 2]*eq.Q + 3*_C[1, 2]*x[2]^2)*y[2]
                    - _D[1, 1]*y[3] - _D[1, 2]*y[4])
        dydt[4] = ( -(_A[2, 1] + _B[2, 1]*eq.Q + 3*_C[2, 1]*x[1]^2)*y[1]
                    -(_A[2, 2] + _B[2, 2]*eq.Q + 3*_C[2, 2]*x[2]^2)*y[2]
                    - _D[2, 1]*y[3] - _D[2, 2]*y[4])
    end

    # add forcing (can be nothing)
    eq.forcing(t, x, dxdt, y, dydt)

    return dydt
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# FORCING TERMS

# no forcing (for the homogeneous equation)
no_forcing(t, x, dxdt, y, dydt) = dydt

# sensitivity ∂f/∂Q
dfdQ_forcing(t, x, dxdt, y, dydt) =
    (@inbounds dydt[3] += -(_B[1, 1]*x[1] + _B[1, 2]*x[2]);
     @inbounds dydt[4] += -(_B[2, 1]*x[1] + _B[2, 2]*x[2]); dydt)