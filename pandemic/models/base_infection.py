import numpy as np
import openmdao.api as om

class BaseInfection(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('truncate', default=True, types=bool)
        
    def setup(self):
        nn = self.options['num_nodes']

        # Params

        self.add_input('beta',
               val = np.zeros(nn))

        self.add_input('sigma',
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

        self.add_output('sigma_sq', np.zeros(nn))

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('theta', ['beta', 'sigma', 't'], rows=arange, cols=arange)
        self.declare_partials('theta', ['a', 't_on', 't_off'])

        self.declare_partials('sigma_sq', ['sigma'], rows=arange, cols=arange)

        self.theta_vec_dirs = ['beta', 'sigma', 't']
        self.theta_scalar_dirs = ['a', 't_on', 't_off']

    def apply_theta_derivs(self, ROC, val, jacobian):   
        """ Multiply by the derivatives of the elements of the filtered
        control vector, theta via chain rule.
        """
        for param in self.theta_vec_dirs:
            jacobian[ROC, param] = val * jacobian['theta', param] 

        for param in self.theta_scalar_dirs:
            jacobian[ROC, param] = np.diag(val * jacobian['theta', param])

    def compute(self, inputs, outputs):
        beta, sigma, a, t_on, t_off, t = inputs['beta'], inputs['sigma'], inputs['a'], inputs['t_on'], inputs['t_off'], inputs['t']
        trunc = self.options['truncate']

        # fix numerical overflow
        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        if trunc:
            d_ton[np.where(d_ton > 1e10)] = 1e10
            d_toff[np.where(d_toff > 1e10)] = 1e10

        y = 1 / (1 + d_ton) * 1 / (1 + d_toff) 

        self.theta = (beta - sigma)*y + (1 - y) * beta

        outputs['sigma_sq'] = sigma**2

        outputs['theta'] = self.theta

    def compute_partials(self, inputs, jacobian):
        beta, sigma, a, t_on, t_off, t = inputs['beta'], inputs['sigma'], inputs['a'], inputs['t_on'], inputs['t_off'], inputs['t']
        trunc = self.options['truncate']

        # fix numerical overflow
        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        if trunc:
            d_ton[np.where(d_ton > 1e10)] = 1e10
            d_toff[np.where(d_toff > 1e10)] = 1e10

        jacobian['theta', 'beta'] = 1.0
        jacobian['theta', 'sigma'] = -1/((1 + d_toff)*(1 + d_ton))
        jacobian['theta', 'a'] = beta*((-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))) - (beta - sigma)*(-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - (beta - sigma)*(t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['theta', 't_on'] = a*beta*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2)
        jacobian['theta', 't_off'] = -a*beta*d_toff/((1 + d_toff)**2*(1 + d_ton)) + a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['theta', 't'] = a*(beta - sigma)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*(beta - sigma)*d_toff/((1 + d_toff)**2*(1 + d_ton)) + beta*(-a*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*d_toff/((1 + d_toff)**2*(1 + d_ton)))

        jacobian['sigma_sq', 'sigma'] = 2.0 * sigma 

if __name__ == '__main__':
    import dymos as dm
    import matplotlib.pyplot as plt

    # test derivatives
    p = om.Problem()
    p.model = om.Group()
    n = 35
    p.model.add_subsystem('test', BaseInfection(num_nodes=n), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['t'] = np.linspace(0, 100, n)
    p.run_model()
    p.check_partials(compact_print=True, method='cs')
