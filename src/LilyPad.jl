module LilyPad

using WaterLily
using StaticArrays

include("util.jl")
include("SemiAdvect.jl")
include("LilyFlow.jl")

"""
    LilyPadSim(dims, uBC, L; Δt=1.5, kwargs...)

Semi-Lagrangian simulation. Accepts the same arguments as `WaterLily.Simulation`.
Returns a `WaterLily.Simulation` backed by a `LilyFlow`.

The timestep `Δt` is the primary method knob: the semi-Lagrangian scheme has no
CFL constraint from advection, so larger values trade accuracy for speed.
"""
LilyPadSim(dims, uBC, L; Δt=1.5, kwargs...) =
    WaterLily.Simulation(dims, uBC, L;
        flow_ctor=(dims, uBC; kw...) -> LilyFlow(dims, uBC; kw...),
        Δt, kwargs...)

export LilyPadSim, LilyBiotSim, LilyFlow
# Re-export WaterLily interface so users only need `using LilyPad`
export sim_step!, sim_time, measure!

"""
    LilyBiotSim(...)

Available when `BiotSavartBCs` is loaded. See `LilyPadBiotExt` for full docstring.
"""
function LilyBiotSim end

end # module LilyPad
