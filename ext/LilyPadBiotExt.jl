module LilyPadBiotExt

using LilyPad
using WaterLily
using BiotSavartBCs

"""
    LilyBiotSim(dims, uBC, L; Δt=1.5, nonbiotfaces=(), fmm=true, mem=Array, kwargs...)

Semi-Lagrangian simulation with Biot-Savart boundary conditions for unbounded external flow.
Combines `LilyFlow` (semi-Lagrangian advection) with `BiotSavartPoisson` (pressure solver).
Accepts the same arguments as `WaterLily.Simulation`.

The timestep `Δt` is the primary method knob: the semi-Lagrangian scheme has no
CFL constraint from advection, so larger values trade accuracy for speed.

- `nonbiotfaces`: tuple of face indices to exclude from Biot-Savart BCs (e.g. `(-2,)`)
- `fmm`: use the Fast Multi-level Method (`true`, default) or tree-sum (`false`)
"""
@inline function LilyPad.LilyBiotSim(dims, uBC, L; Δt=1.5, nonbiotfaces=(), fmm=true, mem=Array, kwargs...)
    WaterLily.Simulation(dims, uBC, L;
        flow_ctor=(dims, uBC; kw...) -> LilyFlow(dims, uBC; kw...),
        pois_ctor=flow -> BiotSavartPoisson(flow; nonbiotfaces, fmm, mem),
        Δt, mem, kwargs...)
end

end # module LilyPadBiotExt
