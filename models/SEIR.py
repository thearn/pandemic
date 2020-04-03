import numpy as np
import openmdao.api as om
try:
    from .ks import KS
    from .SIR import SIR
    from .bootstrap_model import generate_phase, make_plots, setup_and_run_phase
except:
    from ks import KS
    from SIR import SIR
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
# incubation rate (1/days needed to become infectious)
alpha = 1.0 / 5.0

# set up model states
states = {'S' : {'name' : 'susceptible', 'rate_source' : 'Sdot', 
                 'targets' : ['S'], 'defect_scaler' : ds, 
                 'interp_s' : pop_total - initial_exposure, 'interp_f' : 0, 'c' : 'orange'},
          'E' : {'name' : 'exposed', 'rate_source' : 'Edot', 
                 'targets' : ['E'], 'defect_scaler' : ds, 
                 'interp_s' : initial_exposure, 'interp_f' : 0.0, 'c' : 'brown'},
          'I' : {'name' : 'infected', 'rate_source' : 'Idot', 
                 'targets' : ['I'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'green'},
                 }

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma},
          'alpha' : {'targets' : ['alpha'], 'val' : alpha}}

# set up model scalar params
s_params = {'t_on' : {'targets' : ['t_on'], 'val' : 10.0},
            't_off' : {'targets' : ['t_off'], 'val' : 70.0},
            'a' : {'targets' : ['a'], 'val' : 5.0}}

class SEIR(SIR):
    """SEIR epidemiological infection model
       S (suceptible), E (exposed), I (infected), R (recovered).
       
       The new state E represents a non-infectious incubation period determined
       by rate 'alpha'.
    """
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
        super(SEIR, self).compute(inputs, outputs)

        S = inputs['S']
        E = inputs['E']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        theta = self.theta

        outputs['Edot'] = theta * S * I - alpha * E
        outputs['Idot'] = alpha * E - gamma * I

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

        jacobian['max_I', 'I'] = self.dagg_i

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
    p, phase0, traj = generate_phase(SEIR, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True)


    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    # p.driver.opt_settings['mu_init'] = 1.0E-2
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['linear_solver'] = 'mumps'
    p.driver.opt_settings['max_iter'] = 500

    phase0.add_boundary_constraint('I', loc='final', upper=0.01, scaler=1.0)
    
    phase0.add_objective('time', loc='final', scaler=1.0)

    phase0.add_timeseries_output('theta')
    

    setup_and_run_phase(states, p, phase0, traj, t_duration_bounds[0])

    make_plots(states, params)
    plt.show()
