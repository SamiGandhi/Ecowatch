import numpy as np

# Constants
PL_d0 = -40  # Path loss at the reference distance (in dB)
SS_d0 = -30  # Signal strength at the reference distance (in dB)
n = 3        # Path loss exponent (typical for urban areas)
sigma = 5    # Standard deviation for shadowing (in dB)
d0 = 1       # Reference distance (in meters)

def calculate_path_loss(d):
    """
    Calculate the path loss for a given distance.
    """
    return PL_d0 + 10 * n * np.log10(d / d0)

def calculate_shadowing():
    """
    Generate a shadowing effect based on a Gaussian distribution.
    """
    return np.random.normal(0, sigma)

def calculate_signal_strength(d):
    """
    Calculate the signal strength at a given distance.
    """
    path_loss = calculate_path_loss(d)
    shadowing = calculate_shadowing()
    return SS_d0 - path_loss + shadowing

# Example Usage
distance = 5  # Distance between transmitter and receiver in meters
signal_strength = calculate_signal_strength(distance)
print(f"Signal Strength at {distance} meters: {signal_strength} dB")
