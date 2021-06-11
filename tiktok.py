import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spi

class System:
    def __init__(self, num_mets, freq, amplitude, mass):
        self.num_mets = num_mets
        self.met_mass = mass
        self.met_freq = freq
        self.met_amp = amplitude
        self.met_k = freq**2 * mass
        phases = np.random.random(num_mets) * np.pi * 2
        # phases = np.zeros((num_mets,))
        # phases[0] = 0.25 * np.pi * 2
        # array that contains the state (velocity and position) of each metronome
        self.states = np.array([amplitude * freq * np.cos(phases),
                                amplitude * np.sin(phases)]).T
        self.system_mass = mass * self.num_mets

    def step(self, t, states):
        states = np.reshape(states, (len(states)//2, 2))
        res = np.empty_like(states)
        # Total momentum due to the metronomes
        met_net_momentum = np.sum(states[:, 0] * self.met_mass)
        # ensure that momentum of the entire is conserved at zero
        system_vel = met_net_momentum / self.system_mass
        
        # The change in position is equal to the velocity
        res[:, 1] = states[:, 0]

        # The change in velocity is equal to the acceleration
        res[:, 0] = states[:, 1] / -self.met_mass * self.met_k - system_vel

        return res.ravel()
        
    def make_metronomes(num, frequency, amplitude, mass):
        """
        Creates a set of metronomes

        Parameters:
            frequency (float): Frequency of metronome in Hz
            amplitude (float): Max displacement of bob from neutral (m)
            mass      (float): Mass of the bob (kg)
        
        Returns:
            mets (set): set of Metronome objects
        """
        mets = set()
        for _ in range(num):
            mets.add(Metronome(frequency, amplitude, mass))
        return mets 

    def run(self, interval=(0, 10), plot_on=False):
        res = spi.solve_ivp(self.step, interval, self.states.ravel(), max_step=0.005)
        print(res.y.shape)
        if plot_on:
            plt.figure()
            vel_inds = np.arange(self.num_mets * 2)[::2]
            pos_inds = np.arange(self.num_mets * 2)[1::2]
            vels = res.y[vel_inds, :]
            poses = res.y[pos_inds, :]
            
            print(vel_inds)
            print(pos_inds)
            for i in np.arange(self.num_mets):
                plt.plot(res.t, poses[i, :])
            
            plt.plot(res.t, np.sum(vels, axis=0), 'k:', linewidth=2)
            # plt.legend(['velocity', 'position'])
            plt.show()

if __name__ ==  "__main__":
    s = System(10, 1, 1, 1)
    s.run(interval=(0, 20), plot_on=True)