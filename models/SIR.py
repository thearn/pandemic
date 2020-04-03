import numpy as np
import openmdao.api as om
try:
    from .ks import KS
    from .base_infection import BaseInfection
    from .bootstrap_model import generate_phase, make_plots, setup_and_run_phase
except ImportError:
    from ks import KS
    from base_infection import BaseInfection
    from bootstrap_model import generate_phase, make_plots, setup_and_run_phase

# ============== Default configuration =============

pop_total = 1.0
initial_exposure = 0.01 * pop_total
# model discretization 
ns = 50
# defect scaler for model solution
ds = 1e-2

#### VECTOR PARAMS
# baseline contact rate (infectivity)
beta = 0.25
# recovery rate (1/days needed to recover)
gamma = 1.0 / 14.0

# set up model states
states = {'S' : {'name' : 'susceptible', 'rate_source' : 'Sdot', 
                 'targets' : ['S'], 'defect_scaler' : ds, 
                 'interp_s' : pop_total - initial_exposure, 'interp_f' : 0, 'c' : 'orange'},
          'I' : {'name' : 'infected', 'rate_source' : 'Idot', 
                 'targets' : ['I'], 'defect_scaler' : ds, 
                 'interp_s' : initial_exposure, 'interp_f' : pop_total/2, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'green'},
                 }

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma}}

# set up model scalar params
s_params = {'t_on' : {'targets' : ['t_on'], 'val' : 10.0},
            't_off' : {'targets' : ['t_off'], 'val' : 70.0},
            'a' : {'targets' : ['a'], 'val' : 5.0}}


class SIR(BaseInfection):
    """Basic epidemiological infection model
       S (suceptible), I (infected), R (recovered).
    """
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

        self.add_input('gamma',
               val = np.zeros(nn))

        self.add_output('max_I', val=0.0)

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('Sdot', ['beta', 'sigma', 'gamma', 'S', 'I', 'R', 't'], rows=arange, cols=arange)
        self.declare_partials('Sdot', ['a', 't_on', 't_off'])

        self.declare_partials('Idot', ['beta', 'sigma', 'gamma', 'S', 'I', 't'], rows=arange, cols=arange)
        self.declare_partials('Idot', ['a', 't_on', 't_off'])

        self.declare_partials('Rdot', ['gamma', 'I'], rows=arange, cols=arange)

        self.declare_partials('max_I', 'I')

    def compute(self, inputs, outputs):
        super(SIR, self).compute(inputs, outputs)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        theta = self.theta

        # time derivatives of the states of an SIR model
        # substitution dynamic control 'theta' for constant 'beta'.
        outputs['Sdot'] = - theta * S * I
        outputs['Idot'] = theta * S * I - gamma * I
        outputs['Rdot'] = gamma * I

        # aggregate the infection curve to determine peak
        agg_i, self.dagg_i = KS(I)
        outputs['max_I'] = np.sum(agg_i)

    def compute_partials(self, inputs, jacobian):
        super(SIR, self).compute_partials(inputs, jacobian)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        theta = self.theta

        # derivatives of the ODE equations of state
        jacobian['Sdot', 'S'] = -I * theta
        jacobian['Sdot', 'I'] = -S * theta
        # cascade the derivatives w.r.t. theta using the chain rule
        dSdot_dtheta = -I * S
        self.apply_theta_derivs('Sdot', dSdot_dtheta, jacobian)

        jacobian['Idot', 'gamma'] = -I
        jacobian['Idot', 'S'] = I * theta
        jacobian['Idot', 'I'] = S * theta - gamma
        dIdot_dtheta = I * S
        self.apply_theta_derivs('Idot', dIdot_dtheta, jacobian)

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
    p.model.add_subsystem('test', SIR(num_nodes=n, truncate=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)
    p['t'] = np.linspace(0, 100, n)
    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    print()
    raw = input("Continue with baseline sim run test? (y/n)")
    if raw != "y":
        quit()

    p, phase0, traj = generate_phase(SIR, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True)


    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    # p.driver.opt_settings['mu_init'] = 1.0E-2
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['linear_solver'] = 'mumps'
    p.driver.opt_settings['max_iter'] = 500

    phase0.add_boundary_constraint('I', loc='final', upper=0.05, scaler=1.0)
    
    phase0.add_objective('time', loc='final', scaler=1.0)

    phase0.add_timeseries_output('theta')
    
    setup_and_run_phase(states, p, phase0, traj, t_duration_bounds[0])

    make_plots(states, params)

    plt.show()
