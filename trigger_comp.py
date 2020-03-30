import numpy as np
from openmdao.api import ExplicitComponent



# import matplotlib.pyplot as plt

# n=100

# t = np.linspace(0, 10, n)

# default_val = 0.6
# signal = 1/(t**2+1)#np.linspace(0.0, 0.6, n)
# t_trigger = 5.2

# a = 30

# y = 1 / (1 + np.exp(-a*(t - t_trigger)))

# plt.plot(signal*y + (1 - y) * default_val)
# plt.show()
# quit()

class EventTrigger(ExplicitComponent):
    """ Reduces signal?
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('signal',
                       val=0.4*np.ones(nn),
                       desc='signal to be enabled at time t')

        self.add_input('t',
                       val=np.zeros(nn),
                       desc='time vector')

        self.add_input('default_val',
                       val=0.4,
                       desc='default value')        

        self.add_input('t_trigger',
                       val=10.0,
                       desc='trigger time')

        self.add_input('a',
                       val=20.0,
                       desc='scale parameter')

        self.add_output('filtered',
                       val=np.zeros(nn),
                       desc='filtered signal')

        self.add_output('filtered_timescaled',
                       val=np.zeros(nn),
                       desc='filtered signal * time')

        arange = np.arange(self.options['num_nodes'], dtype=int)
        self.declare_partials('filtered', ['t', 'signal'], rows=arange, cols=arange)
        self.declare_partials('filtered', ['t_trigger', 'a', 'default_val'])

        self.declare_partials('filtered_timescaled', ['t', 'signal'], rows=arange, cols=arange)
        self.declare_partials('filtered_timescaled', ['t_trigger', 'a', 'default_val'])


    def compute(self, inputs, outputs):
        t, t_trigger, signal, a, default_val = inputs['t'], inputs['t_trigger'], inputs['signal'], inputs['a'], inputs['default_val']

        y = 1 / (1 + np.exp(-a*(t - t_trigger)))
        filtered = signal*y + (1 - y) * default_val

        outputs['filtered'] = filtered
        outputs['filtered_timescaled'] = (default_val - signal)**2

    def compute_partials(self, inputs, jacobian):
        t, t_trigger, signal, a, default_val = inputs['t'], inputs['t_trigger'], inputs['signal'], inputs['a'], inputs['default_val']

        jacobian['filtered', 't'] = -a*default_val*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2 + a*signal*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2
        jacobian['filtered', 't_trigger'] = a*default_val*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2 - a*signal*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2
        jacobian['filtered', 'signal'] = 1/(1 + np.exp(-a*(t - t_trigger)))
        jacobian['filtered', 'a'] = default_val*(-t + t_trigger)*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2 - signal*(-t + t_trigger)*np.exp(-a*(t - t_trigger))/(1 + np.exp(-a*(t - t_trigger)))**2
        jacobian['filtered', 'default_val'] = 1 - 1/(1 + np.exp(-a*(t - t_trigger)))

        jacobian['filtered_timescaled', 'signal'] = -2*default_val + 2*signal
        jacobian['filtered_timescaled', 'default_val'] = 2*default_val - 2*signal

        ####################

        

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import matplotlib.pyplot as plt

    p = Problem()
    p.model = Group()

    n = 100
    p.model.add_subsystem('test', EventTrigger(num_nodes=n), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)

    p['t'] = t = np.linspace(0, 10, n)
    p['signal'] = signal = np.linspace(0.2, 0.4, n)

    p['default_val'] = 0.4
    p['t_trigger'] = t_trigger = 5.2

    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    filtered = p['filtered']
    filtered_ts = p['filtered_timescaled']

    plt.figure()
    plt.plot(t, filtered, label='filtered via trigger, t=%2.1f' % t_trigger)
    plt.plot(t, signal, label='original')
    plt.legend()

    plt.figure()
    plt.plot(t, filtered_ts)


    plt.show()



