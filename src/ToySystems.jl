module ToySystems

export no_forcing, f_forcing

# //////////////////////////////////////////////
# FORCING FUNCTIONS FOR THE LINEARISED EQUATIONS
# Note that by default these functions add to
# their last inputs, which is the only argument
# that gets modified
# //////////////////////////////////////////////

# no forcing does nothing (for the homogeneous equation)
no_forcing(t, u, dudt, v, dvdt) = (dvdt)

# forcing based on the vector field (note this is a closure)
f_forcing(χ) = wrapped(t, u, dudt, v, dvdt) = (dvdt .+= χ.*dudt; dvdt)

include("ninemodesystem.jl")
include("aeroelastic.jl")
include("lorenz.jl")

end