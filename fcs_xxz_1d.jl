using Distributed
addprocs(60)

@everywhere begin

include(joinpath(dirname(@__DIR__), "tools.jl"))
using ITensors
using JLD2
using ProgressMeter
using Dates

mutable struct Observer <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64
    min_sweep::Int64
    info::AbstractVector

    Observer(ener_tol=1e-12; min_sweep=5) = new(ener_tol, 1000., min_sweep, [])
end

function ITensors.checkdone!(o::Observer;kwargs...)
    sweep  = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy-o.last_energy)/abs(energy) < o.energy_tol && sweep > o.min_sweep
        println("Stopping DMRG after sweep $sweep")
        o.info ∋ (sweep=sweep)
        return true
    end

    o.last_energy = energy
    return false
end

function xxzModel(params)
    (; t, nsites, cutoff, ener_tol) = params
    sites = siteinds("S=1/2", nsites, conserve_qns=true)

    os = OpSum()
    for j=1:nsites-1
      os += t, "Sz",j,"Sz",j+1
      os += 1/2,"S+",j,"S-",j+1
      os += 1/2,"S-",j,"S+",j+1
    end

    H = MPO(os,sites)

    psi0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n=1:nsites])

    nsweeps = 200
    maxdim  = [10,20,100,100,200, 1000]
    cutoff  = [cutoff]
    noise   = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0]
    obs     = Observer(ener_tol)

    ener, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, observer=obs, outputlevel=0)
    return psi
end

function Fcs(params)
    (; psi, λ, x0, x1) = params
    sites = siteinds(psi)
    gates = ITensor[]

    for i in x0:x1
        g_i = exp(im * λ * op("Sz", sites[i]))
        gates ∋ g_i
    end

    state = apply(gates, psi)

    return -log(inner(psi, state))
end

function Region(nsites, len_reg)
    mid = trunc(nsites / 2) |> Int
    half_reg = trunc(len_reg / 2) |> Int
    x0 = mid - half_reg + 1
    x1 = mid + (len_reg - half_reg)
    return Int(x0), Int(x1)
end


filename(name) = joinpath(@__DIR__, name)





end



# cutoff   = 1e-14
# ener_tol = 1e-12
# nsites   = 101
# ts       = [-0.9:0.1:0.9..., 1.2:0.2:3...]
# # ts       = 0.2 * ones(10)
# λs       = range(0, 2π, 201)

# params = [(t=t, nsites=nsites, cutoff=cutoff, ener_tol) for t in ts]

# gstate_s = pmap(xxzModel, params)

# max_dims = maxlinkdim.(gstate_s)
# @show max_dims

# x0, x1 = Region(nsites, 21)

# fcs_s = []
# @showprogress for psi in gstate_s
#     params = [(psi=psi, λ=λ, x0=x0, x1=x1) for λ in λs]
#     fcs_s ∋ pmap(Fcs, params)
# end


# @show fcs_s

# jldsave(filename("data/fcs_xxz_l$(nsites)_reg$(x1-x0+1)_$(today()).jld2"), fcs_s=fcs_s, ts=ts, nsites=nsites, cutoff=cutoff, ener_tol=ener_tol, x0=x0, x1=x1, max_dims=max_dims)


# x0, x1 = Region(nsites, 20)

# fcs_s = []
# @showprogress for psi in gstate_s
#     params = [(psi=psi, λ=λ, x0=x0, x1=x1) for λ in λs]
#     fcs_s ∋ pmap(Fcs, params)
# end


# @show fcs_s

# jldsave(filename("data/fcs_xxz_l$(nsites)_reg$(x1-x0+1)_$(today()).jld2"), fcs_s=fcs_s, ts=ts, nsites=nsites, cutoff=cutoff, ener_tol=ener_tol, x0=x0, x1=x1, max_dims=max_dims)



nsites   = 101
ener_tol = 1e-12
ts       = [0.6, 0.6]
cutoff_s = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-11, 1e-12, 1e-13, 1e-14]
x0, x1   = Region(nsites, 21)
λs       = range(0, 2π, 201)
fcs_s    = []

params = [(t=t, nsites=nsites, cutoff=cutoff, ener_tol) for cutoff in cutoff_s, t in ts]
gstate_s = pmap(xxzModel, params)

max_dims = maxlinkdim.(gstate_s)
@show max_dims

fcs_s = []
@showprogress for psi in gstate_s
    params = [(psi=psi, λ=λ, x0=x0, x1=x1) for λ in λs]
    fcs_s ∋ pmap(Fcs, params)
end

jldsave(filename("data/fcs_xxz_cutoffs_$(today()).jld2"), fcs_s=fcs_s, nsites = nsites, ts=ts, ener_tol=ener_tol, cutoff_s=cutoff_s, x0=x0, x1=x1, max_dims=max_dims)
