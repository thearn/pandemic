import numpy as np
import openmdao.api as om
try:
    from .ks import KS
    from .base_infection import BaseInfection
except:
    from ks import KS
    from base_infection import BaseInfection

class SIR(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        super(SIR, self).setup()

        nn = self.options['num_nodes']

        # States
        self.add_input('S',
               val=np.zeros(nn))

        self.add_input('I',
               val=np.zeros(nn))

        self.add_input('R',
               val=np.zeros(nn))

        # ROCs
        self.add_output('Sdot', val=np.zeros(nn))
        self.add_output('Idot', val=np.zeros(nn))
        self.add_output('Rdot', val=np.zeros(nn))

        # Params
        self.add_input('beta',
               val = np.zeros(nn))
        self.add_input('sigma',
               val = np.zeros(nn))
        self.add_input('gamma',
               val = np.zeros(nn))

        self.add_output('theta',
               val = np.zeros(nn))

        self.add_output('max_I', val=0.0)

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('Sdot', ['beta', 'sigma', 'gamma', 'S', 'I', 'R'], rows=arange, cols=arange)

        self.declare_partials('Idot', ['beta', 'sigma', 'gamma', 'S', 'I'], rows=arange, cols=arange)

        self.declare_partials('Rdot', ['gamma', 'I'], rows=arange, cols=arange)

        self.declare_partials('theta', ['beta', 'sigma'], rows=arange, cols=arange)

        self.declare_partials('max_I', 'I')

    def compute(self, inputs, outputs):
        super(SIR, self).compute(inputs, outputs)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        beta = inputs['beta']
        sigma = inputs['sigma']
        
        theta = beta -  sigma
        outputs['theta'] = theta


        # aggregate the infection curve to determine peak
        agg_i, self.dagg_i = KS(I)
        outputs['max_I'] = np.sum(agg_i)

        # time derivatives of the states of an SIR model
        # substitution dynamic control 'theta' for constant 'beta'.
        outputs['Sdot'] = - theta * S * I
        outputs['Idot'] = theta * S * I - gamma * I
        outputs['Rdot'] = gamma * I

    def compute_partials(self, inputs, jacobian):
        super(SIR, self).compute_partials(inputs, jacobian)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        beta = inputs['beta']
        sigma = inputs['sigma']

        theta = beta - sigma

        # derivatives of the ODE equations of state
        jacobian['theta', 'beta'] = 1.0
        jacobian['theta', 'sigma'] = -1.0

        jacobian['Sdot', 'S'] = -I * theta
        jacobian['Sdot', 'I'] = -S * theta
        jacobian['Sdot', 'beta'] = - S * I
        jacobian['Sdot', 'sigma'] = S * I

        jacobian['Idot', 'gamma'] = -I
        jacobian['Idot', 'S'] = I * theta
        jacobian['Idot', 'I'] = S * theta - gamma
        jacobian['Idot', 'beta'] = S * I
        jacobian['Idot', 'sigma'] = - S * I

        jacobian['Rdot', 'gamma'] = I
        jacobian['Rdot', 'I'] = gamma

        jacobian['max_I', 'I'] = self.dagg_i

if __name__ == '__main__':
    import dymos as dm
    import matplotlib.pyplot as plt

    # test derivatives
    p = om.Problem()
    p.model = om.Group()
    n = 35
    p.model.add_subsystem('test', SIR(num_nodes=n), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)
    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    print()
    print(p['max_I'], np.max(p['I']))
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
    phase = dm.Phase(ode_class=SIR,
                   transcription=dm.GaussLobatto(num_segments=ns, 
                                                 order=3))
    p.model.linear_solver = om.DirectSolver()
    phase.set_time_options(fix_initial=True, duration_bounds=(200.0, 301.0))

    ds = 1e-2
    phase.add_state('S', fix_initial=True, rate_source='Sdot', targets=['S'], lower=0.0,
                  upper=pop_total, ref=pop_total/2, defect_scaler = ds)
    phase.add_state('I', fix_initial=True, rate_source='Idot', targets=['I'], lower=0.0,
                  upper=pop_total, ref=pop_total/2, defect_scaler = ds)
    phase.add_state('R', fix_initial=True, rate_source='Rdot', targets=['R'], lower=0.0,
                  upper=pop_total, ref=pop_total/2, defect_scaler = ds)

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
    p.set_val('traj.phase0.states:I',
            phase.interpolate(ys=[infected0, pop_total/2], nodes='state_input'))
    p.set_val('traj.phase0.states:R',
            phase.interpolate(ys=[0, pop_total/2], nodes='state_input'))

    p.run_driver()
    sim_out = traj.simulate()

    t = sim_out.get_val('traj.phase0.timeseries.time')
    s = sim_out.get_val('traj.phase0.timeseries.states:S')
    i = sim_out.get_val('traj.phase0.timeseries.states:I')
    r = sim_out.get_val('traj.phase0.timeseries.states:R')

    theta = sim_out.get_val('traj.phase0.timeseries.theta')

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.title('baseline simulation - no mitigation')
    plt.plot(t, s/pop_total, 'orange', lw=2, label='susceptible')
    plt.plot(t, i/pop_total, 'teal', lw=2, label='infected')
    plt.plot(t, r/pop_total, 'g', lw=2, label='recovd/immune')
    plt.xlabel('days')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, len(t)*[beta], lw=2, label='$\\beta$')
    plt.plot(t, theta, lw=2, label='$\\theta$(t)')
    plt.legend()
    plt.show()
