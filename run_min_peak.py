import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om 

from pandemic.models.SEIRDS import SEIRDS, states, params, s_params
from pandemic.bootstrap_problem import generate_phase, setup_and_run_phase, make_plots

# population
pop_total = 1.0
initial_exposure = 0.01 * pop_total

# model discretization 
ns = 30
# defect scaler for model solution
ds = 1e-2
# simulated time duration (days)
t_duration = 365.0 / 2

# VECTOR PARAMS
# baseline contact rate (infectivity)
beta = 0.25
# recovery rate (recovery rate / days needed to resolve infection)
gamma = 0.95 / 14.0
# incubation rate (1/days needed to become infectious)
alpha = 1.0 / 5.0
# immunity loss rate (1/days needed to become susceptible again)
epsilon = 1.0 / 365.0
# death rate (mortality rate / days average to resolve infection)
#   should be complementry to recovery (gamma)
mu = 0.05 / 14.0

# SCALAR PARAMS
# control params for mitigation
t_on = 15.0 # on time
t_off = 75.0 # off time
a = 20.0 # smoothness parameter


# set up model states defect scalar
for state in states:
    states[state]['ds'] = ds

# set initial condition, guessed final value
states['S']['interp_s'], states['S']['interp_f'] = 0.995, 0.2
states['E']['interp_s'], states['E']['interp_f'] = 0.005, 0.2
states['I']['interp_s'], states['I']['interp_f'] = 0.0, 0.2
states['R']['interp_s'], states['R']['interp_f'] = 0.0, 0.2
states['D']['interp_s'], states['D']['interp_f'] = 0.0, 0.2 

# set parameter values for run
params['beta']['val'] = beta
params['alpha']['val'] = alpha
params['gamma']['val'] = gamma
params['epsilon']['val'] = epsilon
params['mu']['val'] = mu
                 

t_initial_bounds = [0.0, 0.0]
t_duration_bounds = [t_duration, t_duration]

# set model scalar params
s_params['t_on']['val'] = t_on
s_params['t_off']['val'] = t_off
s_params['a']['val'] = a

p, phase0, traj = generate_phase(SEIRDS, ns, states, params, s_params, 
                                 t_initial_bounds, t_duration_bounds, 
                                 fix_initial=True, fix_duration=True)


p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.options['print_results'] = False
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# p.driver.opt_settings['mu_init'] = 1.0E-2
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.opt_settings['max_iter'] = 500
p.driver.opt_settings['tol'] = 1e-10


# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.opt_settings['iSumm'] = 6


#phase0.add_boundary_constraint('I', loc='final', upper=0.01, scaler=1.0)

# contact rate mitigation between t_on and t_off
phase0.add_control('sigma', targets=['sigma'], lower=0.0, upper=beta, ref=beta)

# minimize infection curve
phase0.add_objective('max_I', scaler=1e2)

# record effective contact rate
phase0.add_timeseries_output('theta')

setup_and_run_phase(states, p, phase0, traj, t_duration)

# plot all states
fig = make_plots(states, params)

max_I = np.max(states['I']['result'])

fig.suptitle('peak infection = %2.2f by mitigation between %2.2f and %2.2f' % (max_I, t_on, t_off))
plt.show()



