import numpy as np
import openmdao.api as om
try:
    from .ks import KS
    from .base_infection import BaseInfection
except:
    from ks import KS
    from base_infection import BaseInfection
    from SIR import SIR

class SEIR(SIR):

    def setup(self):
        super(SEIR, self).setup()

        nn = self.options['num_nodes']

        # New states
        self.add_input('E',
               val=np.zeros(nn))

        # new ROCs
        self.add_output('Edot', val=np.zeros(nn))

        # New params
        self.add_input('alpha',
               val = np.zeros(nn))

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('Edot', ['alpha', 'beta', 'sigma', 'gamma', 'S', 'E', 'I', 'R', 't'], rows=arange, cols=arange)
        self.declare_partials('Edot', ['a', 't_on', 't_off'])

        self.declare_partials('Idot', ['alpha', 'E'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # want the baseinfection compute, but not the SIR one
        super(SIR, self).compute(inputs, outputs)

        S = inputs['S']
        E = inputs['E']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        theta = self.theta

        # aggregate the infection curve to determine peak
        agg_i, self.dagg_i = KS(I)
        outputs['max_I'] = np.sum(agg_i)

        outputs['Sdot'] = - theta * S * I
        outputs['Edot'] = theta * S * I - alpha * E
        outputs['Idot'] = alpha * E - gamma * I
        outputs['Rdot'] = gamma * I

    def compute_partials(self, inputs, jacobian):
        # want the baseinfection jacobian, but not the SIR one
        super(SIR, self).compute_partials(inputs, jacobian)

        S = inputs['S']
        E = inputs['E']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        theta = self.theta

        # derivatives of infection curve aggregation
        jacobian['max_I', 'I'] = self.dagg_i

        # derivatives of the ODE equations of state
        jacobian['Sdot', 'S'] = - I * theta
        jacobian['Sdot', 'I'] = - S * theta
        # cascade the derivatives w.r.t. theta using the chain rule
        dSdot_dtheta = -I * S
        self.apply_theta_derivs('Sdot', dSdot_dtheta, jacobian)

        jacobian['Edot', 'S'] = I * theta
        jacobian['Edot', 'I'] = S * theta
        jacobian['Edot', 'alpha'] = - E
        jacobian['Edot', 'E'] = - alpha
        dEdot_dtheta = S * I
        self.apply_theta_derivs('Edot', dEdot_dtheta, jacobian)

        jacobian['Idot', 'alpha'] = E
        jacobian['Idot', 'E'] = alpha
        jacobian['Idot', 'gamma'] = - I
        jacobian['Idot', 'I'] = - gamma

        jacobian['Rdot', 'gamma'] = I
        jacobian['Rdot', 'I'] = gamma

if __name__ == '__main__':
    import dymos as dm
    import matplotlib.pyplot as plt

    # test derivatives
    p = om.Problem()
    p.model = om.Group()
    n = 35
    p.model.add_subsystem('test', SEIR(num_nodes=n, truncate=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['E'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)
    p['alpha'] = np.random.uniform(0, 2, n)
    p['t'] = np.linspace(0, 100, n)
    
    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    print()
    raw = input("Continue with baseline sim run test? (y/n)")
    if raw != "y":
        quit()
    # test baseline model
    pop_total = 1.0
    infected0 = 0.01
    ns = 50

    p = om.Problem(model=om.Group())
    traj = dm.Trajectory()

    p.model.add_subsystem('traj', subsys=traj)
    phase = dm.Phase(ode_class=SEIR,
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

    p.driver = om.pyOptSparseDriver()

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    #p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
    #p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
    p.driver.opt_settings['iSumm'] = 6

    p.driver.declare_coloring() 

    alpha = 1.0 / 5.0
    beta = 0.25
    gamma = 0.95 / 14.0

    phase.add_input_parameter('alpha', targets=['alpha'], dynamic=True, val=alpha)
    phase.add_input_parameter('beta', targets=['beta'], dynamic=True, val=beta)
    phase.add_input_parameter('gamma', targets=['gamma'], dynamic=True, val=gamma)

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
            phase.interpolate(ys=[0, pop_total/2], nodes='state_input'))
    p.set_val('traj.phase0.states:R',
            phase.interpolate(ys=[0, pop_total/2], nodes='state_input'))

    p.run_driver()
    sim_out = traj.simulate()

    t = sim_out.get_val('traj.phase0.timeseries.time')
    s = sim_out.get_val('traj.phase0.timeseries.states:S')
    e = sim_out.get_val('traj.phase0.timeseries.states:E')
    i = sim_out.get_val('traj.phase0.timeseries.states:I')
    r = sim_out.get_val('traj.phase0.timeseries.states:R')

    theta = sim_out.get_val('traj.phase0.timeseries.theta')

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.title('baseline simulation - no mitigation')
    plt.plot(t, s/pop_total, 'orange', lw=2, label='susceptible')
    plt.plot(t, e/pop_total, 'red', lw=2, label='exposed')
    plt.plot(t, i/pop_total, 'teal', lw=2, label='infected')
    plt.plot(t, r/pop_total, 'g', lw=2, label='recovd/immune')
    plt.xlabel('days')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, len(t)*[beta], lw=2, label='$\\beta$')
    plt.plot(t, theta, lw=2, label='$\\theta$(t)')
    plt.legend()
    plt.show()
