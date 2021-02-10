# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# load packages
using StochasticDiffEq, DiffEqCallbacks
using Plots
using Statistics
using LinearAlgebra
using Random
using DiffEqNoiseProcess
using LaTeXStrings
#################################################
# parameters
numtrajplot = 256 # .. for plotting

# time range for the solver
dt = 0.0001f0 #0.001f0
tinterval = 0.02f0
tstart = 0.0f0
Nintervals = 150 # total number of intervals, total time = t_interval*Nintervals
tspan = (tstart,tinterval*Nintervals)
ts = Array(tstart:dt:(Nintervals*tinterval+dt)) # time array for noise grid

# Hamiltonian parameters
Δ = 20.0f0
Ωmax = 10.0f0 # control parameter (maximum amplitude)
κ = 1.0f0

# loss hyperparameters
C1 = Float32(1.0)  # evolution state fidelity
C2 = Float32(0.0) # action amplitudes
C3 = Float32(0.0) # evolution state fidelity for last few steps!

struct Parameters{flType,intType,tType}
	numtrajplot::intType
	dt::flType
	tinterval::flType
	tspan::tType
	Nintervals::intType
	ts::Vector{flType}
	Δ::flType
	Ωmax::flType
	κ::flType
	C1::flType
	C2::flType
	C3::flType
end

myparameters = Parameters{typeof(dt),typeof(numtrajplot), typeof(tspan)}(
  numtrajplot, dt, tinterval, tspan, Nintervals, ts,
  Δ, Ωmax, κ, C1, C2, C3)

###############################################
# strategy

function control(u, Ωmax)
  ceR, cdR, ceI, cdI = u

  if (cdI*ceR-cdR*ceI) > 0
   Ω = Ωmax
  else
	Ω = -Ωmax
  end
  return Ω
end

###############################################
# initial state anywhere on the Bloch sphere
function prepare_initial(dt, n_par)
  # shape 4 x n_par
  # input number of parallel realizations and dt for type inference
  # random position on the Bloch sphere
  theta = acos.(2*rand(typeof(dt),n_par).-1)  # uniform sampling for cos(theta) between -1 and 1
  phi = rand(typeof(dt),n_par)*2*pi  # uniform sampling for phi between 0 and 2pi
  # real and imaginary parts ceR, cdR, ceI, cdI
  u0 = [cos.(theta/2), sin.(theta/2).*cos.(phi), false*theta, sin.(theta/2).*sin.(phi)]
  return vcat(transpose.(u0)...) # build matrix
end

# target state
# ψtar = |up>

u0 = prepare_initial(myparameters.dt, myparameters.numtrajplot)

###############################################
# Define SDE

function qubit_drift!(du,u,p,t)
  # expansion coefficients |Ψ> = ce |e> + cd |d>
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  # Δ: atomic frequency
  # Ω: Rabi frequency for field in x direction
  # κ: spontaneous emission
  Δ, Ωmax, κ = p[1:3]
  Ω = control(u, Ωmax)

  @inbounds begin
    du[1] = 1//2*(ceI*Δ-ceR*κ+cdI*Ω)
    du[2] = -cdI*Δ/2 + 1*ceR*(cdI*ceI+cdR*ceR)*κ+ceI*Ω/2
    du[3] = 1//2*(-ceR*Δ-ceI*κ-cdR*Ω)
    du[4] = cdR*Δ/2 + 1*ceI*(cdI*ceI+cdR*ceR)*κ-ceR*Ω/2
  end
  return nothing
end

function qubit_diffusion!(du,u,p,t)
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  κ = p[end]

  du .= false

  @inbounds begin
    #du[1] = zero(ceR)
    du[2] += sqrt(κ)*ceR
    #du[3] = zero(ceR)
    du[4] += sqrt(κ)*ceI
  end
  return nothing
end

# normalization callback
condition(u,t,integrator) = true
function affect!(integrator)
  integrator.u=integrator.u/norm(integrator.u)
end
cb = DiscreteCallback(condition,affect!,save_positions=(false,false))


CreateGrid(t,W1) = NoiseGrid(t,W1)

# set scalar random process
W = sqrt(myparameters.dt)*randn(typeof(myparameters.dt),size(myparameters.ts)) #for 1 trajectory
W1 = cumsum([zero(myparameters.dt); W[1:end-1]], dims=1)
NG = CreateGrid(myparameters.ts,W1)

# get control pulses
p_all = [myparameters.Δ; myparameters.Ωmax; myparameters.κ]
# define SDE problem
prob = SDEProblem{true}(qubit_drift!, qubit_diffusion!, vec(u0[:,1]), myparameters.tspan, p_all,
   callback=cb, noise=NG
   )

#########################################
# visualization -- run for new batch
function visualize(u0, prob::SDEProblem, myparameters::Parameters;
	 alg=EulerHeun(), all_traj = true
	 )
  pars = [myparameters.Δ; myparameters.Ωmax; myparameters.κ]

  function prob_func(prob, i, repeat)
    # prepare initial state and applied control pulse
	u0tmp = deepcopy(vec(u0[:,i]))
	W = sqrt(myparameters.dt)*randn(typeof(myparameters.dt),size(myparameters.ts)) #for 1 trajectory
    W1 = cumsum([zero(myparameters.dt); W[1:end-1]], dims=1)
    NG = CreateGrid(myparameters.ts,W1)

    remake(prob,
	    p = pars,
	  	u0 = u0tmp,
  	  	callback = cb,
		noise=NG
		)
  end

  ensembleprob = EnsembleProblem(prob,
   prob_func = prob_func,
   safetycopy = true
   )

  u = solve(ensembleprob, alg, EnsembleThreads(),
 	saveat=myparameters.tinterval,
 	dt=myparameters.dt,
 	adaptive=false, #abstol=1e-6, reltol=1e-6,
 	trajectories=myparameters.numtrajplot, batch_size=myparameters.numtrajplot)


  ceR = @view u[1,:,:]
  cdR = @view u[2,:,:]
  ceI = @view u[3,:,:]
  cdI = @view u[4,:,:]
  infidelity = @. (cdR^2 + cdI^2) / (ceR^2 + cdR^2 + ceI^2 + cdI^2)
  meaninfidelity = mean(infidelity)
  loss = myparameters.C1*meaninfidelity

  @info "Loss: " loss

  fidelity = @. (ceR^2 + ceI^2) / (ceR^2 + cdR^2 + ceI^2 + cdI^2)

  mf = mean(fidelity, dims=2)[:]
  sf = std(fidelity, dims=2)[:]

  # re-compute actions
  arrayu = Array(u)
  Ωlist = []
  for i = 1:(size(arrayu)[2])
	Ω = [control(arrayu[:,i,j], myparameters.Ωmax) for j = 1:(size(arrayu)[3])]
    push!(Ωlist, Ω)
  end
  Ωlist = hcat(Ωlist...)
  ma = mean(Ωlist, dims=1)[:]
  sa = std(Ωlist, dims=1)[:]
  #@show size(Ωlist) size(sa)

  pl1 = plot(0:myparameters.Nintervals, mf,
	  ribbon = sf,
	  ylim = (0,1), xlim = (0,myparameters.Nintervals),
	  c=:navyblue, lw = 1.5, xlabel = L"i", title=L"F(t_i)", legend=false, grid=false)
  pl2 = plot(0:myparameters.Nintervals, ma,
	  ribbon = sa,
	  ylim=(-myparameters.Ωmax,myparameters.Ωmax), xlim = (0,myparameters.Nintervals),
	  c=:orangered4, lw = 1.5, xlabel = L"i", title=L"\Omega(t_i)", legend=false, grid=false)
  if all_traj
     plot!(pl1, fidelity, legend=false, c=:gray, alpha=0.1)
	 plot!(pl2, Ωlist', legend=false, c=:gray, alpha=0.1)
  else
     plot!(pl1, 0:myparameters.Nintervals, fidelity[:,end],  c=:black, lw = 1.5, legend=false)
	 plot!(pl2, 0:myparameters.Nintervals, Ωlist[end,:], c=:black, lw = 1.5, legend=false)
  end

  pl = plot(pl1, pl2, layout = (1, 2), legend = false, size=(800,360))
  return pl1, pl2, loss, infidelity
end

u0 = prepare_initial(myparameters.dt, myparameters.numtrajplot)
u0[1,end] = 0.0f0
u0[2,end] = 1.0f0
u0[3,end] = 0.0f0
u0[4,end] = 0.0f0
pl1, pl2, lossval1, infid = @time visualize(u0, prob, myparameters, all_traj=false)

@info 1-mean(infid)
@info std(infid)

# pl = plot(pl1, pl2, margin=3.0Plots.mm, legend = false, size=(600,300))
# savefig(pl,"Figures/hand_crafted.png")

exit()
