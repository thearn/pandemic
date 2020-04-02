import numpy as np
import openmdao.api as om
try:
  from .ks import KS
except:
  from ks import KS

class SEIRD(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # States
        self.add_input('S',
                       val=np.zeros(nn))

        self.add_input('E',
                       val=np.zeros(nn))

        self.add_input('I',
                       val=np.zeros(nn))

        self.add_input('R',
                       val=np.zeros(nn))

        self.add_input('D',
                       val=np.zeros(nn))

        # ROCs
        self.add_output('Sdot', val=np.zeros(nn))
        self.add_output('Edot', val=np.zeros(nn))
        self.add_output('Idot', val=np.zeros(nn))
        self.add_output('Rdot', val=np.zeros(nn))
        self.add_output('Ddot', val=np.zeros(nn))

        # Params
        self.add_input('alpha',
                       val = np.zeros(nn))

        self.add_input('beta',
                       val = np.zeros(nn))

        self.add_input('sigma',
                       val = np.zeros(nn))

        self.add_input('gamma',
                       val = np.zeros(nn))

        self.add_input('epsilon',
                       val = np.zeros(nn))

        self.add_input('mu',
                       val = np.zeros(nn))

        self.add_input('t',
                       val = np.zeros(nn))

        self.add_input('a',
                       val=5.0,
                       desc='scale parameter')
        self.add_input('t_on',
                       val=20.0,
                       desc='trigger time')
        self.add_input('t_off',
                       val=60.0,
                       desc='trigger time')

        self.add_output('theta',
                       val=np.zeros(nn))
        self.add_output('max_I', 0.0)

        self.add_output('sigma_sq', np.zeros(nn))

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('Sdot', ['beta', 'sigma', 'epsilon', 'S', 'I', 'R', 't'], rows=arange, cols=arange)
        self.declare_partials('Sdot', ['a', 't_on', 't_off'])

        self.declare_partials('Edot', ['beta', 'sigma', 'S', 'E', 'I', 't', 'alpha'], rows=arange, cols=arange)
        self.declare_partials('Edot', ['a', 't_on', 't_off'])

        self.declare_partials('Idot', ['gamma', 'E', 'I', 'alpha', 'mu'], rows=arange, cols=arange)
        self.declare_partials('Rdot', ['gamma', 'epsilon', 'I', 'R'], rows=arange, cols=arange)

        self.declare_partials('Ddot', ['mu', 'I'], rows=arange, cols=arange)

        self.declare_partials('theta', ['beta', 'sigma', 't'], rows=arange, cols=arange)
        self.declare_partials('theta', ['a', 't_on', 't_off'])

        self.declare_partials('sigma_sq', ['sigma'], rows=arange, cols=arange)
        self.declare_partials('max_I', 'I')

    def compute(self, inputs, outputs):
        beta, sigma, mu, epsilon, gamma, S, E, I, R, a, t_on, t_off, t, alpha = inputs['beta'], inputs['sigma'], inputs['mu'], inputs['epsilon'], inputs['gamma'], inputs['S'], inputs['E'], inputs['I'], inputs['R'], inputs['a'], inputs['t_on'], inputs['t_off'], inputs['t'], inputs['alpha']
        
        # determine a cut-off where the infection is gone
        I[np.where(I < 1e-4)] = 0.0
        #E[np.where(E < 1e-6)] = 0.0

        # fix numerical overflow
        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        d_ton[np.where(d_ton > 1.e10)] = 1.e10
        d_toff[np.where(d_toff > 1.e10)] = 1.e10

        y = 1 / (1 + d_ton) * 1 / (1 + d_toff) 

        theta = (beta - sigma)*y + (1 - y) * beta

        agg_i, self.dagg_i = KS(I)
        outputs['max_I'] = np.sum(agg_i)

        outputs['sigma_sq'] = sigma**2

        outputs['theta'] = theta

        outputs['Sdot'] = -theta * S * I + epsilon * R
        outputs['Edot'] = theta * S * I - alpha * E
        outputs['Idot'] = alpha * E - gamma * I - mu * I
        outputs['Rdot'] = gamma * I - epsilon * R
        outputs['Ddot'] = mu * I

    def compute_partials(self, inputs, jacobian):
        beta, sigma, mu, epsilon, gamma, S, E, I, R, a, t_on, t_off, t, alpha = inputs['beta'], inputs['sigma'], inputs['mu'], inputs['epsilon'], inputs['gamma'], inputs['S'], inputs['E'], inputs['I'], inputs['R'], inputs['a'], inputs['t_on'], inputs['t_off'], inputs['t'], inputs['alpha']
        
        # determine a cut-off where the infection is gone
        I[np.where(I < 1e-4)] = 0.0
        #E[np.where(E < 1e-6)] = 0.0

        # fix numerical overflow
        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        d_ton[np.where(d_ton > 1.e10)] = 1.e10
        d_toff[np.where(d_toff > 1.e10)] = 1.e10

        jacobian['Sdot', 'beta'] = -I*S
        jacobian['Sdot', 'sigma'] = I*S/((1 + d_toff)*(1 + d_ton))
        jacobian['Sdot', 'epsilon'] = R
        jacobian['Sdot', 'S'] = I*(-beta*(1 - 1/((1 + d_toff)*(1 + d_ton))) - (beta - sigma)/((1 + d_toff)*(1 + d_ton)))
        jacobian['Sdot', 'I'] = S*(-beta*(1 - 1/((1 + d_toff)*(1 + d_ton))) - (beta - sigma)/((1 + d_toff)*(1 + d_ton)))
        jacobian['Sdot', 'R'] = epsilon
        jacobian['Sdot', 'a'] = I*S*(-beta*((-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))) + (beta - sigma)*(-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (beta - sigma)*(t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton)))
        jacobian['Sdot', 't_on'] = I*S*(-a*beta*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2))
        jacobian['Sdot', 't_off'] = I*S*(a*beta*d_toff/((1 + d_toff)**2*(1 + d_ton)) - a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)))
        jacobian['Sdot', 't'] = I*S*(-a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)) - beta*(-a*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*d_toff/((1 + d_toff)**2*(1 + d_ton))))

        jacobian['Edot', 'beta'] = I*S
        jacobian['Edot', 'sigma'] = -I*S/((1 + d_toff)*(1 + d_ton))
        jacobian['Edot', 'S'] = I*(beta*(1 - 1/((1 + d_toff)*(1 + d_ton))) + (beta - sigma)/((1 + d_toff)*(1 + d_ton)))
        jacobian['Edot', 'E'] = -alpha
        jacobian['Edot', 'I'] = S*(beta*(1 - 1/((1 + d_toff)*(1 + d_ton))) + (beta - sigma)/((1 + d_toff)*(1 + d_ton)))
        jacobian['Edot', 'a'] = I*S*(beta*((-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))) - (beta - sigma)*(-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - (beta - sigma)*(t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton)))
        jacobian['Edot', 't_on'] = I*S*(a*beta*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2))
        jacobian['Edot', 't_off'] = I*S*(-a*beta*d_toff/((1 + d_toff)**2*(1 + d_ton)) + a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)))
        jacobian['Edot', 't'] = I*S*(a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)) + beta*(-a*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*d_toff/((1 + d_toff)**2*(1 + d_ton))))
        jacobian['Edot', 'alpha'] = -E

        jacobian['Idot', 'mu'] = -I
        jacobian['Idot', 'gamma'] = -I
        jacobian['Idot', 'E'] = alpha
        jacobian['Idot', 'I'] = -gamma - mu
        jacobian['Idot', 'alpha'] = E

        jacobian['Rdot', 'gamma'] = I
        jacobian['Rdot', 'epsilon'] = -R
        jacobian['Rdot', 'I'] = gamma
        jacobian['Rdot', 'R'] = -epsilon

        jacobian['Ddot', 'mu'] = I
        jacobian['Ddot', 'I'] = mu


        jacobian['theta', 'beta'] = 1.0
        jacobian['theta', 'sigma'] = -1/((1 + d_toff)*(1 + d_ton))
        jacobian['theta', 'a'] = beta*((-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))) - (beta - sigma)*(-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - (beta - sigma)*(t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['theta', 't_on'] = a*beta*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2)
        jacobian['theta', 't_off'] = -a*beta*d_toff/((1 + d_toff)**2*(1 + d_ton)) + a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['theta', 't'] = a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)) + beta*(-a*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*d_toff/((1 + d_toff)**2*(1 + d_ton)))

        jacobian['max_I', 'I'] = self.dagg_i

        jacobian['sigma_sq', 'sigma'] = 2.0 * sigma

if __name__ == '__main__':
  import dymos as dm
  import matplotlib.pyplot as plt

  # test derivatives
  p = om.Problem()
  p.model = om.Group()
  n = 35
  p.model.add_subsystem('test', SEIRD(num_nodes=n), promotes=['*'])
  p.setup(force_alloc_complex=True)
  np.random.seed(0)
  p['S'] = np.random.uniform(1, 1000, n)
  p['E'] = np.random.uniform(1, 1000, n)
  p['I'] = np.random.uniform(1, 1000, n)
  p['R'] = np.random.uniform(1, 1000, n)
  p['D'] = np.random.uniform(1, 1000, n)

  p['beta'] = np.random.uniform(0, 2, n)
  p['sigma'] = np.random.uniform(0, 2, n)
  p['gamma'] = np.random.uniform(0, 2, n)
  p['t'] = np.linspace(0, 100, n)
  p.run_model()
  p.check_partials(compact_print=True, method='cs')

  print()
  input("Press enter to continue with baseline sim run test...")

  # test baseline model
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

  p.driver = om.pyOptSparseDriver()

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
