import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt

from models import SEIRD

pop_total = 1.0
infected0 = 0.01
ns = 50

p = om.Problem(model=om.Group())

traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)

phase = dm.Phase(ode_class=SEIRD,
                 transcription=dm.GaussLobatto(num_segments=ns, 
                                               order=3))
p.model.linear_solver = om.DirectSolver()
phase.set_time_options(fix_initial=True, duration_bounds=(200.0, 301.0), targets=['t'])
#phase.set_time_options(fix_initial=True, fix_duration=True)


ds = 1e-2
phase.add_state('S', fix_initial=True, rate_source='Sdot', targets=['S'], lower=0.0,
                upper=pop_total, ref=pop_total/2, defect_scaler = ds)
phase.add_state('E', fix_initial=True, rate_source='Edot', targets=['E'], lower=0.0,
                upper=pop_total, ref=pop_total/2, defect_scaler = ds)
phase.add_state('I', fix_initial=True, rate_source='Idot', targets=['I'], lower=0.0,
                upper=pop_total, ref=pop_total/2, defect_scaler = ds)
phase.add_state('R', fix_initial=True, rate_source='Rdot', targets=['R'], lower=0.0,
                upper=pop_total, ref=pop_total/2, defect_scaler = ds)
phase.add_state('D', fix_initial=True, rate_source='Ddot', targets=['D'], lower=0.0,
                upper=pop_total, ref=pop_total/2, defect_scaler = ds)
phase.add_state('int_sigma', rate_source='sigma_sq', lower=0.0, defect_scaler = 1e-2)

#p.driver = om.ScipyOptimizeDriver()

p.driver = om.pyOptSparseDriver()
#p.driver.options['optimizer'] = 'SNOPT'
#p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
#p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
#p.driver.opt_settings['iSumm'] = 6


p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# p.driver.opt_settings['mu_init'] = 1.0E-2
p.driver.opt_settings['nlp_scaling_method'] = 'user-scaling'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['linear_solver'] = 'mumps'

p.driver.declare_coloring() 


beta = 0.25
gamma = 0.95 / 14.0
alpha = 1.0 / 5.0
epsilon = 1.0 / 365.
mu = (1 - 14*gamma) / 14.0
lim = 0.15

phase.add_input_parameter('alpha', targets=['alpha'], dynamic=True, val=alpha)
phase.add_input_parameter('beta', targets=['beta'], dynamic=True, val=beta)
phase.add_input_parameter('gamma', targets=['gamma'], dynamic=True, val=gamma)
phase.add_input_parameter('epsilon', targets=['epsilon'], dynamic=True, val=epsilon)
phase.add_input_parameter('mu', targets=['mu'], dynamic=True, val=mu)

# just converge ODEs
phase.add_objective('time', loc='final')

phase.add_timeseries_output('theta')


traj.add_phase(name='phase0', phase=phase)
p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', 200)

p.set_val('traj.phase0.states:S',
          phase.interpolate(ys=[pop_total - infected0, 0], nodes='state_input'))
p.set_val('traj.phase0.states:E',
          phase.interpolate(ys=[infected0, 0], nodes='state_input'))
p.set_val('traj.phase0.states:I',
          phase.interpolate(ys=[0, pop_total/3], nodes='state_input'))
p.set_val('traj.phase0.states:R',
          phase.interpolate(ys=[0, pop_total/3], nodes='state_input'))
p.set_val('traj.phase0.states:D',
          phase.interpolate(ys=[0, pop_total/3], nodes='state_input'))

p.run_driver()
sim_out = traj.simulate()

t = sim_out.get_val('traj.phase0.timeseries.time')
s = sim_out.get_val('traj.phase0.timeseries.states:S')
e = sim_out.get_val('traj.phase0.timeseries.states:E')
i = sim_out.get_val('traj.phase0.timeseries.states:I')
r = sim_out.get_val('traj.phase0.timeseries.states:R')
d = sim_out.get_val('traj.phase0.timeseries.states:D')

int_sigma = sim_out.get_val('traj.phase0.timeseries.states:int_sigma')
print("objective:", int_sigma[-1])

theta = sim_out.get_val('traj.phase0.timeseries.theta')

fig = plt.figure(figsize=(10, 8))
plt.title('baseline simulation - no mitigation')
plt.subplot(511)
plt.plot(t, s, 'orange', lw=2, label='susceptible')
plt.legend(), plt.xticks(np.arange(0, t[-1], 50), " ")

plt.subplot(512)
plt.plot(t, e, 'k', lw=2, label='exposed')
plt.legend(), plt.xticks(np.arange(0, t[-1], 50), " ")

plt.subplot(513)
plt.plot(t, i, 'teal', lw=2, label='infected')
plt.legend(), plt.xticks(np.arange(0, t[-1], 50), " ")

plt.subplot(514)
plt.plot(t, r, 'g', lw=2, label='recovd/immune')
plt.legend(), plt.xticks(np.arange(0, t[-1], 50), " ")

plt.subplot(515)
plt.plot(t, d, lw=2, label='dead')

plt.xlabel('days')
plt.legend()

fig = plt.figure(figsize=(10, 5))
plt.subplot(211)
print("dead:", d[-1])
plt.title('baseline simulation - no mitigation')
plt.plot(t, s/pop_total, 'orange', lw=2, label='susceptible')
plt.plot(t, e/pop_total, 'k', lw=2, label='exposed')
plt.plot(t, i/pop_total, 'teal', lw=2, label='infected')
plt.plot(t, r/pop_total, 'g', lw=2, label='recovd/immune')
plt.plot(t, d/pop_total, lw=2, label='dead')
plt.xlabel('days')
plt.legend()
plt.subplot(212)
plt.plot(t, len(t)*[beta], lw=2, label='$\\beta$')
plt.plot(t, theta, lw=2, label='$\\theta$(t)')
plt.legend()
plt.show()