module LilyPad

using WaterLily
using StaticArrays

include("util.jl")
include("SemiAdvect.jl")
include("SemiMomStep.jl")


# LilyPadSim: thin wrapper around WaterLily.Simulation that replaces mom_step!
# with semi_mom_step!. Property access falls through to the inner simulation,
# so sim.flow, sim.U, sim.L, sim_time(sim), etc. all work unchanged.

"""
    LilyPadSim(args...; fixed_dt=nothing, kwargs...)

Semi-Lagrangian simulation wrapper. Accepts exactly the same arguments as
`WaterLily.Simulation`; the only behavioural difference is that `sim_step!`
calls `semi_mom_step!` instead of `WaterLily.mom_step!`.

Set `fixed_dt` to use a constant timestep (legacy LilyPad-style stepping).
Leave `fixed_dt=nothing` to keep CFL-updated stepping.
"""
struct LilyPadSim <: WaterLily.AbstractSimulation
    sim :: WaterLily.Simulation
    fixed_dt :: Union{Nothing,Float64}
    LilyPadSim(args...; fixed_dt=nothing, kwargs...) =
        new(WaterLily.Simulation(args...; kwargs...),
            isnothing(fixed_dt) ? nothing : Float64(fixed_dt))
end

Base.getproperty(s::LilyPadSim, name::Symbol) =
    name === :sim ? getfield(s, :sim) :
    name === :fixed_dt ? getfield(s, :fixed_dt) :
    getproperty(s.sim, name)
Base.setproperty!(s::LilyPadSim, name::Symbol, x) =
    name === :sim ? setfield!(s, :sim, x) :
    name === :fixed_dt ? setfield!(s, :fixed_dt, isnothing(x) ? nothing : Float64(x)) :
    setproperty!(s.sim, name, x)

"""
    sim_step!(sim::LilyPadSim; remeasure=true, kwargs...)

Advance `sim` by one time step using the semi-Lagrangian momentum step.
"""
function WaterLily.sim_step!(sim::LilyPadSim; remeasure=true, kwargs...)
    remeasure && measure!(sim)
    if isnothing(sim.fixed_dt)
        semi_mom_step!(sim.flow, sim.pois; kwargs...)
    else
        semi_mom_step!(sim.flow, sim.pois;
                       step_dt=convert(eltype(sim.flow.p), sim.fixed_dt),
                       update_cfl=false,
                       kwargs...)
    end
end

export LilyPadSim
# Re-export WaterLily interface so users only need `using LilyPad`
export sim_step!, sim_time, measure!

end # module LilyPad
