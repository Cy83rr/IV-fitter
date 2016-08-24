import glob
import os

import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

sampleDimension = 6
echarge = 1
epsilonZero = 8.854187817
epsilon = 3.9
tox = 285
cox = epsilon * epsilonZero / tox *1e3


def model(voltages, rcontact, n0, vdirac, mobility):
    return 2 * rcontact + (sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility))

initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=1e3, min=1e2, max=2e4)
initialParameters.add('rcontact', value=1e4, min=1e3, max=1e6)
initialParameters.add('n0', value=1e10, min=1e8, max=1e12)
initialParameters.add('vdirac', value=60, min=55, max=65)

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(os.path.curdir, '*.txt')):
    with open(infile) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        currents = numpy.array(data[:, 1])
        currentsErr = numpy.array(data[:, 2])
        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])

        # fitting to data using leastsq method
        gmod = lmfit.Model(model)
        result = gmod.fit(resistance, voltages=voltages, params=initialParameters)

        # check if directory exists, create if needed
        resultPath=os.path.curdir+'/results/'
        directory = os.path.dirname(resultPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # saving fit result to a text file
        resultName = os.path.split(os.path.splitext(infile)[0])[1]
        with open(os.path.join(resultPath, resultName + '_Fit.txt'), "w+") as fitResult:
            fitResult.write(result.fit_report())
        # TODO remove
        print(lmfit.fit_report(result, min_correl=0.1))
        pyplot.figure()
        scatter = pyplot.scatter(voltages, resistance)
        initFitLine, = pyplot.plot(voltages, result.init_fit, 'k--')
        bestFitLine, = pyplot.plot(voltages, result.best_fit, 'r-')
        pyplot.xlabel('Gate voltage [ V ]')
        pyplot.ylabel('Resistance [ M\u2126 ]')
        pyplot.text(-8, 167, 'Sample dimension [length/width]: ' + str(sampleDimension))
        pyplot.text(-8, 152, 'V_DS: 10 V')
        pyplot.title('Charakterystyka przejsciowa')
        pyplot.legend([scatter, initFitLine, bestFitLine], ['Data', 'Initial Fit', 'Best Fit'], loc='upper left')
        pyplot.savefig(os.path.join(resultPath, resultName))
