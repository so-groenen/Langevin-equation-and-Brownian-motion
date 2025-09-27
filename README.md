# Solving the Langevin equation

This python notebook contains notes, explicit calculations and numerical results for the Langevin equations.<br>
It is quite old and needs some cleanup.<br>
Summary: 
## Mean square displacement
The Langevin equation
$$\frac{d v}{dt} = -\frac{v}{\tau_{bath}} + v_T\sqrt{\frac{2}{\tau_{bath}}} \eta(t),\qquad \text{where }v_T=\sqrt{\frac{k_BT}{m}}, \quad\langle \eta(t) \rangle = 0,\qquad \langle \eta(t)\eta(t') \rangle = \delta(t-t')$$
where $\eta(t)$ is a delta correlated Gaussian white noise with mean $0$ and the prefactor describes the fluctuation amplitude. As thermal equilibrium is built in by construction, dissipation, controlled by the bath correlation time $\tau_B$ is inside the fluctuation term. 

We are mainly interested in the mean square displacement, where the average is unerstood as ensemble average (N simulations):
In particular, the two limiting case are the ballistic limit at $t < \tau$, when the particle has some initial inertia and has some memory from where it comes from, and the diffusive limit, where all the initial energy has been dissipated into the bath, and the particle is driven by the (diffusive) Brownian motion. Energy lost into the bath (dissipation), via the dissipative term is regained by the thermal fluctuations (fluctuation-dissipation theorem).
$$ |\Delta x(t)|^2 := \Big\langle (x(t) -x_0)^2 \Big\rangle = \frac{1}{N}\sum^N_n [x_n(t) - x_n(0)]^2 $$
whose solution is

$$
 |\Delta x(t)|^2 = \begin{cases}
   v_0^2 t^2 ,& \text{if } t\ll \tau \\
    2 \frac{k_B T}{m} \tau t =: 2Dt,              & t \gg \tau \text{   fluctuation=dissipation}
\end{cases} $$
<center><img src="langevin_results/Mean_square_displacement_Brownian_Diffusion_coeff.jpg" width="800"></center>

## Brownian diffusion: Thermal equilibrium ($v_0 = v_T$) and $t\gg \tau_B$

The Kubo-Green formula links the (dissipative) diffusion constant to the (fluctuating) velocity correlator at thermal equilibrium:

$$ D = \lim_{t\to \infty} \frac{|\Delta r(t)|^2}{2t}  = \int_0^\infty dt \Big\langle v(t)v(0) \Big\rangle = v_T^2 \tau$$

We can use it to numerically infer the diffusion constant (using the same heat bath correlation times):

<center><img src="langevin_results/Green_Kubo_Brownian_Diffusion_coeff.jpg" width="800"></center>

## Ultra short timescales $t\ll \tau_B$: Ballistic regime

Let us look at even shorter timescales. We set $dt = 0.00001s$ ($10^{-5} s$) to get a better resolution:
<center><img src="langevin_results/Ballistic regime.jpg" width="800"></center>

## Thermalization: Out-of-Equilibrium initial conditions $v_0 < v_T$

Let us have a look at the Langevin equation outside of thermal equilibrium, ie, we start with particles NOT thermalized to the bath. 
We prepare a cold particles with initial velocity $v_0 = \sqrt{\frac{k_B T_0}{m}}$, that is, they are thermalized with a temperature $T_0 < T_B$ lower than the heat bath temperature $T_B$. How fast to they 'speed up' and thermalize with the bath? We are interested in
$$\frac{1}{2}k_B T(t) = \frac{1}{2} m \langle  v(t)^2\rangle $$


<center><img src="langevin_results/Thermalization of a cold particle in a heat bath in 1d.jpg" width="800"></center>
