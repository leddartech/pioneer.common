class SaturationCalibration():
    # TODO: document when the final form has converged.

    def __init__(self, distance_coeff:float, amplitude_coeffs:tuple):
        self.distance_coeff = distance_coeff
        self.amplitude_coeffs = amplitude_coeffs

    def __call__(self, plateau_size):
        distance = self.distance_coeff*plateau_size
        amplitude = self.amplitude_coeffs[0]*np.exp(self.amplitude_coeffs[1]*plateau_size)
        return distance, amplitude
