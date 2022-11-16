---
title: FVM solver for compressible flow
summary: This is a sovler simulating a 2D compressible fluid flow with Finite Volume Method. 
tags:
  - Simulation
date: '2022-9-3T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: Photo of the distribution of velocity
  focal_point: Smart

# links:
#   - icon: github
#     icon_pack: fab
#     name: Follow
#     url: https://github.com/Howw-Way/MSRA/tree/master/Torch101
url_code: ''
url_pdf: ''
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---

<!-- Author: Howw -->
<!-- Data: 22.8.30 -->

# Description

This is my notes for simulation a 2D compressible fluid flow with Finite Volume Method. Original version is written by Philip Mocz[1], respect to him for his work and the spirit of open-source.

Generally speaking, FVM discretizes the partial differential equations with finite volume in tiny girds. The conservation equations are derived in each volume, so the variables are naturally conserved in each volume. The variables change in time and space, and be calculated with the flux in each face of a volume. While the flux in each face is reconstructed using the variables stored in each grid.

**Reference:**
[1][URL](https://levelup.gitconnected.com/create-your-own-finite-volume-fluid-simulation-with-python-8f9eab0b8305)


## Variables 

### Primitive Form
- Density $\rho$
- Velocity $v_x$,$v_y$
- Pressure $p$

### Conservative Form

- Mass Density $\rho$
- Momentum Density $\rho v_x$,$\rho v_y$
- Pressure $p$

Obviously, those two form can switch to each other, in this code function `getConserved` and `getPrimitive` proceed the switch. 

## Equations 

### State equations
$$p=(\gamma -1) \rho u \tag{1.1}$$

$$c=\sqrt{\gamma \frac{p}{\rho}} \tag{1.2}$$

$$e=u+\frac{v_x^2+v_y^2}{2} \tag{1.3}$$

In which $c$ is the local soundspeed, $\gamma$ is the ideal gas adiabatic index parameter (a monatomic ideal gas has $\gamma=5/3$), $u$ is the internal energy(related to temperature), $e$ is the total energy,  $v_x^2$ and $v_y^2$ are the velocity of fluid.

### Control equation

#### Partial Differential Equation:
$$\frac{\partial U}{\partial t}+ \frac{\partial F_1(U)}{\partial x}+\frac{\partial F_2(U)}{\partial y}=0 \tag{2.1}$$
In which: 
$$U=\begin{pmatrix}
 \rho\\
 \rho v_x\\
 \rho v_y\\
\rho e
\end{pmatrix}, 
F_1(U)=\begin{pmatrix}
 \rho v_x\\
 \rho v_x^2+p\\
 \rho v_x v_y\\
(\rho e +p)v_x
\end{pmatrix},
F_2(U)=\begin{pmatrix}
 \rho v_y\\
 \rho v_x v_y\\
 \rho v_y^2+p\\
(\rho e +p)v_y
\end{pmatrix}
$$

In which the $U$ is the matrix of conservative variable, $F(U)$ is the flux function. 

#### Integrated equation:

$$\frac{\partial U}{\partial t}+\frac{1}{\Omega_{i,j}}\oint_{\partial \Omega}{F\cdot \vec{n} }ds=0 \tag{2.2}$$

In which $\Omega$ is the area of a grid, for 2D problem, $\Omega=\Delta x $ or $\Delta y$, for 3D problem, $\Omega=\Delta x \Delta y $, and $\vec{n}$ is the normal vector of face. 

#### Discretization

During computation of FVM, the fluid is discretized into individual fluid elements (square ‘cells’) of size $\Delta x \Delta y \Delta x$ . The cells exchange conservative quantities via fluxes though cell interfaces.

For time discretization, the forward difference is proceed.

$$\frac{U_i^{n+1}-U_i^n}{\Delta t}+\frac{1}{\Omega_{i,j}}\oint_{\partial \Omega}{F\cdot \vec{n} }ds=0 \tag{2.3}$$

Assuming $Q_i=U_i \cdot (\Delta x)^2$
$$\to Q^{n+1}_i=Q^{n}_i-\Delta t \Delta x\sum_{j} \hat{F}_{ij}^{n+\frac{1}{2}} \tag{2.4}$$

In which $\hat{F}_{ij}^{n+\frac{1}{2}}$ refers to the numerical flux between neighboring cells i and j, $\Delta x$ can be regard as the area of interface of two grids

In general, the calculation of flux is the key process of FVM, and it is often calculated as a function of the fluid variables on the ‘Left’ and ‘Right’ side of the interface. 

## Calculation of flux

### Extrapolation in Space

For FVM method, all variables are stored at the cell-centered, we need use those variables to calculate(extrapolate) the unknown variables at a distance $Δx/2$ from the cell center to a face. 

Taking spatially extrapolating from a cell $(i,j)$ to the face $(i+1/2,j)$ on its right as an example, the calculation is accomplished as:

$$f_{i+\frac{1}{2},j} \simeq f_{i,j}+\frac{\partial f_{i,j}}{\partial x} \cdot \frac{\Delta x}{2} \tag{2.5}$$

In this code, function `extrapolateInSpaceToFace` performed spatial extrapolation on an arbitrary field to each of the 4 faces of a cell.

It turns out that in general it is **better to extrapolate primitive variables then convert back to conservative**, rather than extrapolate conservative variables directly, in order to ensure the pressure does not accidentally get reconstructed to negative values due to truncation errors.

### Calculating and Applying Fluxes by Rusanov flux

$$\hat{F}=\frac{1}{2}(F_L+F_R)-\frac{c_{max}}{2}(U_R-U_L) \tag{2.6}$$

The first term is a simple average of the fluxes as derived from the left or the right fluid variables. Then, there is an added term which creates numerical diffusivity. It keeps the solution numerically stable. $c_{max}=max({c_i+|v_i|})$ is the maximum signal speed. Advanced versions of flux solvers exist which solve strong shock structures more accurately with less numerical diffusivity, but for our purposes here the Rusanov flux will suffice.

In this code, it is done by function `getFlux`.

### Calculating conserved fluid quantities
Once the fluxes are computed, they can be applied to the conserved fluid quantities $Q$ in each cell. 

In this code, it is done by function `applyFluxes`

## Time Stepping

For numerical stability and accuracy, the simulation timestep cannot be arbitrarily large. It must obey the Courant-Friedrichs-Lewy (CFL) condition:

$$\Delta t=CFL \cdot min\frac{\Delta x}{c_i+|v_i|} \tag{3.1}$$

where $CFL \le 1 $, the speed $c_i+|v_i|$ is a proxy for the maximum signal speed in a cell. 

Conceptually, what the CFL condition says is that in the duration of a timestep, the max signal speed may not travel more than the length of a cell.

## Initial Condition
In this probelm, the domain is assumed to be 2-dimensional and periodic. 

The code specifies the initial primitive variables (density, velocity, pressure fields), and the ideal gas $\gamma$ parameter. 

To set up the Kelvin-Helmholtz instability, the code initializes a high-density region moving to the right and the background moving to the left. 

Pressure is uniform. A small perturbation in the velocity directed perpendicularly to this shear at the interface boundaries is added to induce the instability.

## Pipeline of the algorithm

- Get cell-centered primitive variables from conservative variables (for the first step, it's from the initial condition)`getPrimitive`

- Calculate the next timestep Δt based on Eq. 3.1

- Calculate gradients of primitive variables `getGradient`

- Extrapolate primitive variables in time by Δt/2 using gradients

- Extrapolate primitive variables to faces using gradients `extrapolateInSpaceToFace`

- Feed in face Left and Right fluid states to compute the fluxes across each face `getFlux`

- Update the solution by applying fluxes to the conservative variables `applyFluxes`


OK, that's all for intro, let's take a look at the code

```python
import numpy as np
import matplotlib.pyplot as plt

def getConserved( rho, vx, vy, P, gamma, vol ):
	"""
    Calculate the conserved variable(Mass, Momentumx, Momentumy, Energy) from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	gamma    is ideal gas gamma
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	"""
	Mass   = rho * vol
	Momx   = rho * vx * vol
	Momy   = rho * vy * vol
	Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
	
	return Mass, Momx, Momy, Energy

def getPrimitive( Mass, Momx, Momy, Energy, gamma, vol ):
	"""
    Calculate the primitive variable(rho, vx, vy, P) from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	gamma    is ideal gas gamma
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol
	P   = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
	
	return rho, vx, vy, P

def getGradient(f, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
	f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dx)

    #Info about roll(): it rolls the elements of matrix
	# with the roll, the f_dx is the center difference
	
	return f_dx, f_dy

def slopeLimit(f, dx, f_dx, f_dy):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	
	return f_dx, f_dy

def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """
    Perform spatial extrapolation on an arbitrary field to each of the 4 faces of a cell
    Purpose: To look up fluid variables at on the ‘Left’ and ‘Right’ sides of cell faces for the flux calculation
    Eq. 2.5
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    dx       is the cell size
    f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
    f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
    f_YR     is a matrix of spatial-extrapolated values on `left' face along y-axis 
    f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
    """
    # directions for np.roll() 
    R = -1   # right
    L = 1    # left
    
    f_XL = f - f_dx * dx/2
    #[?]why there need roll?
    f_XL = np.roll(f_XL,R,axis=0)
    f_XR = f + f_dx * dx/2
    
    f_YL = f - f_dy * dx/2
    f_YL = np.roll(f_YL,R,axis=1)
    f_YR = f + f_dy * dx/2
    
    return f_XL, f_XR, f_YL, f_YR

def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
	Eq. 2.6 and Eq. 2.2 (each element of factor in Eq. 2.2 is calculated here)
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	P_L          is a matrix of left-state  pressure
	P_R          is a matrix of right-state pressure
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
	en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)

	# compute star (averaged) states
	rho_star  = 0.5*(rho_L + rho_R)
	momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
	en_star   = 0.5*(en_L + en_R)
	
	P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2/rho_star + P_star
	flux_Momy   = momx_star * momy_star/rho_star
	flux_Energy = (en_star+P_star) * momx_star/rho_star
	
	# find wavespeeds
	C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
	C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
	C = np.maximum( C_L, C_R )
	
	# add stabilizing diffusive term
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
	"""
    This function is uesd for the time stepping, Eq.2.4
	F        is a matrix of the conserved variable field
	flux_F_X is a matrix of the x-dir fluxes
	flux_F_Y is a matrix of the y-dir fluxes
	dx       is the cell size
	dt       is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	F += - dt * dx * flux_F_X
	F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
	F += - dt * dx * flux_F_Y
	F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)

	#Explanation of the above update operation
	# cc=aa/copy()
	# F += - dt * dx * flux_F_X
	# F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
	# the same as :
	# cc[0,:]=aa[0,:]-aa[-1,:]
	# cc[1:,:]=aa[1:,:]-aa[0:-1,:]

	# F += - dt * dx * flux_F_Y
	# F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)
	# the same as :
	# cc[:,0]=aa[:,0]-aa[:,-1]
	# cc[:,1:]=aa[:,1:]-aa[:,0:-1]
	
	return F

def main():
	""" Finite Volume simulation """
	
	# Simulation parameters
	N                      = 128 # resolution
	boxsize                = 1.
	gamma                  = 5/3 # ideal gas gamma
	courant_fac            = 0.4
	t                      = 0
	tEnd                   = 2
	tOut                   = 0.02 # draw frequency
	useSlopeLimiting       = False
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Mesh
	dx = boxsize / N
	vol = dx**2
	xlin = np.linspace(0.5*dx, boxsize-0.5*dx, N)
	Y, X = np.meshgrid( xlin, xlin )
	
	# Generate Initial Conditions - opposite moving streams with perturbation
	w0 = 0.1
	sigma = 0.05/np.sqrt(2.)
	rho = 1. + (np.abs(Y-0.5) < 0.25)
	vx = -0.5 + (np.abs(Y-0.5)<0.25)
	vy = w0*np.sin(4*np.pi*X) * ( np.exp(-(Y-0.25)**2/(2 * sigma**2)) + np.exp(-(Y-0.75)**2/(2*sigma**2)) )
	P = 2.5 * np.ones(X.shape)

	# Get conserved variables
	Mass, Momx, Momy, Energy = getConserved( rho, vx, vy, P, gamma, vol )
	
	# prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	outputCount = 1
	
	# Simulation Main Loop
	while t < tEnd:
		
		# get Primitive variables
		rho, vx, vy, P = getPrimitive( Mass, Momx, Momy, Energy, gamma, vol )
		
		# get time step (CFL) = dx / max signal speed
		dt = courant_fac * np.min( dx / (np.sqrt( gamma*P/rho ) + np.sqrt(vx**2+vy**2)) )
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			dt = outputCount*tOut - t
			plotThisTurn = True
		
		# calculate gradients
		rho_dx, rho_dy = getGradient(rho, dx)
		vx_dx,  vx_dy  = getGradient(vx,  dx)
		vy_dx,  vy_dy  = getGradient(vy,  dx)
		P_dx,   P_dy   = getGradient(P,   dx)
		
		# slope limit gradients
		if useSlopeLimiting:
			rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
			vx_dx,  vx_dy  = slopeLimit(vx , dx, vx_dx,  vx_dy )
			vy_dx,  vy_dy  = slopeLimit(vy , dx, vy_dx,  vy_dy )
			P_dx,   P_dy   = slopeLimit(P  , dx, P_dx,   P_dy  )
		
		# extrapolate half-step in time
		rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
		vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + (1/rho) * P_dx )
		vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + (1/rho) * P_dy )
		P_prime   = P   - 0.5*dt * ( gamma*P * (vx_dx + vy_dy)  + vx * P_dx + vy * P_dy )
		
		# extrapolate in space to face centers
		rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
		vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  dx)
		vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  dx)
		P_XL,   P_XR,   P_YL,   P_YR   = extrapolateInSpaceToFace(P_prime,   P_dx,   P_dy,   dx)
		
		# compute fluxes (local Lax-Friedrichs/Rusanov)
		flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma)
		flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)
		
		# update solution
		Mass   = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
		Momx   = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
		Momy   = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
		Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
		
		# update time
		t += dt
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and plotThisTurn) or (t >= tEnd):
			plt.cla()
			plt.imshow(rho.T)
			plt.clim(0.8, 2.2)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.001)
			outputCount += 1
			
	
	# Save figure
	plt.savefig('finitevolume.png',dpi=240)
	plt.show()
	    
	return 0

if __name__== "__main__":
  main()

```

And here shows the result.

![](./figure/figure.gif)