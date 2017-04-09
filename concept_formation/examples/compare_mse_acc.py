import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

#acuity = 1/(2 * sqrt(pi))
#acuity = 0.001
acuity = 1/sqrt(2*pi)
#acuity = 0.21

def compute(std):
    #mse = std * std
    ecg1 = 1/(2 * sqrt(pi)) * (1 / std)
    ecg2 = 1/(2 * sqrt(pi)) * (1 / max(std, 1/(2*sqrt(pi))))
    ecg3 = 1/(2 * sqrt(pi)) * (1 / sqrt(std*std + 1/(4*pi)))
    
    #ecg = 1 / (sqrt(2*pi) * max(std,acuity))
    #ecg = 1 / (sqrt(2*pi) * (std + acuity))
    #ecg = 1 / sqrt(2 * pi) * 1 / max(std_est, acuity)
    #ecg = (1 - sqrt(ecg)) * (1 - sqrt(ecg))

    return ecg1, ecg2, ecg3

if __name__ == "__main__":

    stds = np.arange(0.0001,1.1001, 0.01)
    ecgs1 = []
    ecgs2 = []
    ecgs3 = []
    for std in stds:
        ecg1, ecg2, ecg3 = compute(std)
        ecgs1.append(ecg1)
        ecgs2.append(ecg2)
        ecgs3.append(ecg3)

    original, = plt.plot(stds, ecgs1)
    acuity, = plt.plot(stds, ecgs2)
    noise, = plt.plot(stds, ecgs3)

    plt.ylim(0,1.3)
    plt.xlim(0,1.0)
    plt.title("Comparison of Cobweb/3 Numeric Correct Guess Formulations")
    plt.xlabel("Standard Deviation of Values ($\sigma$)")
    plt.ylabel("Expected Number of Correct Guesses")
    #plt.legend([original, acuity, noise], ["Unbounded", "Acuity Bounded",
    #                                       "Noisy Estimate"])
    plt.legend([original, acuity, noise], ["$\\frac{1}{\sigma}$", 
                                           "$\\frac{1}{max(\sigma, \sigma_{Acuity})}$",
                                           "$\\frac{1}{\sqrt{\sigma^2 + \sigma_{Acuity}^2}}$"])
    plt.show()

    X = np.arange(0,2,0.01)

    X2 = np.sqrt(X*X + 1/(4*pi))
    s, = plt.plot(X,X)
    ns, = plt.plot(X,X2)
    plt.title("Comparison of Original and Noisy $\sigma$")
    plt.xlabel("Standard Deviation of Values ($\sigma$)")
    plt.legend([s, ns], ["$\sigma$", "$\sqrt{\sigma^2 + \sigma_{Acuity}}$"])
    plt.show()
