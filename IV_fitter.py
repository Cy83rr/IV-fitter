import lmfit
import numpy
import pandas

sampleDimension = 6.0
echarge = 1.6021766208 * 1e-19
epsilonZero = 8.854187817
epsilon = 3.9
tox = 285.0
cox = epsilon * epsilonZero / tox * 1e-3


def model(voltages, rcontact, n0, vdirac, mobility):
    return 2 * rcontact + (sampleDimension / (numpy.sqrt((n0 * echarge)**2 + (cox * (voltages - vdirac))**2) * mobility))


initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=1e3, min=1e2, max=2e4)
initialParameters.add('rcontact', value=1e5, min=1e2, max=1e8)
initialParameters.add('n0', value=1e10, min=1e8, max=1e12)
initialParameters.add('vdirac', value=60, min=55, max=65)

# Iterate over every file with .txt extension
with open('testData1.txt') as dataFile:
    data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
    voltages = numpy.array(data[:, 0])
    currents = numpy.array(data[:, 1])
    currentsErr = numpy.array(data[:, 2])
    resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)]) / 1e5

    # fitting to data using leastsq method
    gmod = lmfit.Model(model)
    result = gmod.fit(resistance, voltages=voltages, params=initialParameters)
    print(lmfit.fit_report(result, min_correl=0.1))
