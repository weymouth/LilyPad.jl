import WaterLily: BC!, mom_project!, CFL, exitBC!, @loop, inside_u, inside, ∂, loc, μddn

"""
    BDIM!(flow)

Apply BDIM blending assuming `flow.f` already stores an absolute advected
velocity-like field.
"""
function BDIM!(flow::Flow)
    @loop flow.f[Ii] -= flow.V[Ii] over Ii ∈ CartesianIndices(flow.f)
    @loop flow.u[Ii] = μddn(Ii, flow.μ₁, flow.f) + flow.V[Ii] + flow.μ₀[Ii] * flow.f[Ii] over Ii ∈ inside_u(size(flow.p))
end

"""
    predict_advect!(flow, dt)

Overwrite `flow.f` with the semi-Lagrangian self-advection of `flow.u⁰`
by applying `scalar_advect!` component-wise.
"""
function predict_advect!(flow::Flow{D,T}, dt) where {D,T}
    flow.f .= flow.u⁰
    for i in 1:D
        scalar_advect!(selectdim(flow.f, D+1, i), selectdim(flow.u⁰, D+1, i), flow.u⁰, dt, i)
    end
end

"""
    correct_advect!(flow, dt)

Overwrite `flow.f` with the semi-Lagrangian advection of `u⁰-0.5dt*μ₀*∇p` under the action of
the old and predicted velocity fields. The 0.5-weighted pressure gradient term at x⁰ is complemented
by the same weighting of the pressure gradient at x after projection, completing the second-order correction.
"""
function correct_advect!(flow::Flow{D,T}, dt) where {D,T}
    flow.f .= flow.u⁰
    flow.u .= (flow.u .+ flow.u⁰) ./ 2 # average velocity for RK2 departure point calculation
    for i in 1:D
        @loop flow.σ[I] = flow.u⁰[I,i] over I ∈ CartesianIndices(flow.σ)
        @loop flow.σ[I] -= (dt / 2) * flow.μ₀[I, i] * ∂(i, I, flow.p) over I ∈ inside(flow.σ)
        scalar_advect!(selectdim(flow.f, D+1, i), flow.σ, flow.u, dt, i)
    end
end

"""
    semi_mom_step!(flow, pois; step_dt=nothing, update_cfl=true, kwargs...)

Advance `flow` by one time step with a legacy-style semi-Lagrangian
predictor/corrector, then update the time-step estimate via `CFL`.

No body forcing or viscous diffusion in v0.1 (inviscid semi-Lagrangian).

Set `step_dt` to force a fixed timestep for this step. Set `update_cfl=false`
to append the same `step_dt` to `flow.Δt` (legacy fixed-`dt` mode).
"""
@fastmath function semi_mom_step!(flow, pois; step_dt=nothing, update_cfl=true, kwargs...)
    flow.u⁰ .= flow.u
    dt = isnothing(step_dt) ? flow.Δt[end] : step_dt
    flow.Δt[end] = dt
    t₁ = sum(flow.Δt)

    # --- Predictor ---
    @log "p"
    predict_advect!(flow, dt)
    BDIM!(flow)
    BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)
    flow.exitBC && exitBC!(flow.u, flow.u⁰, dt)
    mom_project!(flow, pois, 1, t₁)

    # --- Corrector ---
    @log "c"
    correct_advect!(flow, dt)
    BDIM!(flow)
    BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)
    mom_project!(flow, pois, 0.5, t₁)
    BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)

    push!(flow.Δt, update_cfl ? CFL(flow) : dt)
end
