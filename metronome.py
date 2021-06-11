import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Metronome:
    def __init__(self, frequency, amplitude, mass)
        """
        Metronome Object
        This is modeled as a simple spring mass system without damping. 

        Parameters: 
            frequency (float): Frequency of metronome in Hz
            amplitude (float): Max displacement of bob from neutral (m)
            mass      (float): Mass of the bob (kg)

        """
        # Frequency in rad/s
        self.freq = frequency
        # Amplitude of displacement (from neutral) in m
        self.amp = amplitude
        # Mass of bob (kg)
        self.mass = mass
        # Effective Spring Rate (N/m)
        self.k = self.freq**2 * self.mass
        # phase offset in radians
        self.phase = np.random.random() * np.pi *  2

    def get_spring_force(self, t, state):
        f_spring = -self.amp * self.k * np.cos(self.freq * t + phase)
        return f_spring


class System:
    def __init__(self, metronomes):
        self.num_mets = len(metronomes)
        self.mets = metronomes
        self.vel = 0
        self.mass = metronomes[0].mass * self.num_mets

    def step(self, t):
        # Add up the net momentum of each individual metronome
        met_net_momentum = 0
        for met in self.mets:
            met_net_momentum += met.get_momentum(t)
        # set the velocity of the platform to ensure that the net momentum 
        # of the entire system is conserved at zero
        self.vel = -met_net_momentum / self.mass

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

    