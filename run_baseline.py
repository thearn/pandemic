import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om 

from pandemic.models.SEIRDS import SEIRDS, states, params, s_params
from pandemic.bootstrap_problem import generate_phase, setup_and_run_phase, make_plots

# test baseline model
pop_total = 1.0
initial_exposure = 0.01 * pop_total
# model discretization 
ns = 50
# defect scaler for model solution
ds = 1e-2

#### VECTOR PARAMS
# baseline contact rate (infectivity)
beta = 0.25
# recovery rate (recovery rate / days needed to resolve infection)
gamma = 0.95 / 14.0
# incubation rate (1/days needed to become infectious)
alpha = 1.0 / 5.0
# immunity loss rate (1/days needed to become susceptible again)
epsilon = 1.0 / 300.0
# death rate (mortality rate / days average to resolve infection)
#   should be complementry to recovery (gamma)
mu = 0.05 / 14.0

# set up model states
for state in states:
    states[state]['ds'] = ds

params['beta']['val'] = beta
params['alpha']['val'] = alpha
params['gamma']['val'] = gamma
params['epsilon']['val'] = epsilon
params['mu']['val'] = mu

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

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

phase0.add_boundary_constraint('I', loc='final', upper=0.01, scaler=1.0)

#phase0.add_control('sigma', targets=['sigma'], lower=0.0, upper=beta, ref=beta)
#phase0.add_objective('max_I', scaler=1000.0)

phase0.add_objective('time', loc='final', scaler=1.0)

phase0.add_timeseries_output('theta')

setup_and_run_phase(states, p, phase0, traj, 200.0)

# plot all states
fig = make_plots(states, params)

max_I = np.max(states['I']['result'])

fig.suptitle('peak infection = %2.2f, no mitigation' % max_I)
plt.savefig("images/Figure_1.png")

plt.show()


