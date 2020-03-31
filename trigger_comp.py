import numpy as np
from openmdao.api import ExplicitComponent



import matplotlib.pyplot as plt

# n=300

# t = np.linspace(0, 50, n)

# default_val = 0.6
# signal = (0.05*t)**2
# t_on = 10
# t_off = 30

# a = 10

# y = 1 / (1 + d_ton) * 1 / (1 + np.exp(-a*(t_off - t)))

# plt.subplot(211)
# plt.plot(t, y)
# plt.subplot(212)
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

        self.add_input('t_on',
                       val=20.0,
                       desc='trigger time')

        self.add_input('t_off',
                       val=60.0,
                       desc='trigger time')

        self.add_input('a',
                       val=5.0,
                       desc='scale parameter')

        self.add_output('filtered',
                       val=np.zeros(nn),
                       desc='filtered signal')

        self.add_output('filtered_timescaled',
                       val=np.zeros(nn),
                       desc='filtered signal * time')

        arange = np.arange(self.options['num_nodes'], dtype=int)
        self.declare_partials('filtered', ['t', 'signal'], rows=arange, cols=arange)
        self.declare_partials('filtered', ['t_on', 't_off', 'a', 'default_val'])

        self.declare_partials('filtered_timescaled', ['t', 'signal'], rows=arange, cols=arange)
        self.declare_partials('filtered_timescaled', ['t_on','t_off', 'a', 'default_val'])


    def compute(self, inputs, outputs):
        t, t_on, t_off, signal, a, default_val = inputs['t'], inputs['t_on'], inputs['t_off'], inputs['signal'], inputs['a'], inputs['default_val']

        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        d_ton[np.where(d_ton > 1.e10)] = 1.e10
        d_toff[np.where(d_toff > 1.e10)] = 1.e10

        y = 1 / (1 + d_ton) * 1 / (1 + d_toff) 

        filtered = signal*y + (1 - y) * default_val

        outputs['filtered'] = filtered
        outputs['filtered_timescaled'] = (default_val - signal)**2

        #print(np.sum(outputs['filtered_timescaled'] ))

    def compute_partials(self, inputs, jacobian):
        t, t_on, t_off, signal, a, default_val = inputs['t'], inputs['t_on'], inputs['t_off'], inputs['signal'], inputs['a'], inputs['default_val']

        d_ton = np.exp(-a*(t - t_on))
        d_toff = np.exp(-a*(-t + t_off))

        d_ton[np.where(d_ton > 1.e10)] = 1.e10
        d_toff[np.where(d_toff > 1.e10)] = 1.e10

        jacobian['filtered', 't'] = a*signal*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*signal*d_toff/((1 + d_toff)**2*(1 + d_ton)) + default_val*(-a*d_ton/((1 + d_toff)*(1 + d_ton)**2) + a*d_toff/((1 + d_toff)**2*(1 + d_ton)))
        jacobian['filtered', 't_on'] = a*default_val*d_ton/((1 + d_toff)*(1 + d_ton)**2) - a*signal*d_ton/((1 + d_toff)*(1 + d_ton)**2)
        jacobian['filtered', 't_off'] = -a*default_val*d_toff/((1 + d_toff)**2*(1 + d_ton)) + a*signal*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['filtered', 'signal'] = 1/((1 + d_toff)*(1 + d_ton))
        jacobian['filtered', 'a'] = default_val*((-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) + (t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))) - signal*(-t + t_on)*d_ton/((1 + d_toff)*(1 + d_ton)**2) - signal*(t - t_off)*d_toff/((1 + d_toff)**2*(1 + d_ton))
        jacobian['filtered', 'default_val'] = 1 - 1/((1 + d_toff)*(1 + d_ton))

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

    p['t'] = t = np.linspace(0, 100, n)
    p['signal'] = signal = np.linspace(0.2, 0.4, n)

    p['default_val'] = 0.4
    p['t_on'] = t_on = 5.2

    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    filtered = p['filtered']
    filtered_ts = p['filtered_timescaled']

    plt.figure()
    plt.plot(t, filtered, label='filtered via trigger, t=%2.1f' % t_on)
    plt.plot(t, signal, label='original')
    plt.legend()

    plt.figure()
    plt.plot(t, filtered_ts)


    plt.show()


