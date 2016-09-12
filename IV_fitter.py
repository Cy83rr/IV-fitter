import glob
import logging
import os

import matplotlib.pyplot as pyplot
import numpy
import pandas
from scipy.optimize import curve_fit


# TODO write all necessary dependencies or smth like gradle in java -
# TODO http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html

#############
# Constants
#############

# sample length/width, unitless - default value
sampleDimension = 6
# electron charge in [C]
echarge = 1.6021766208 * 1e-19
#echarge = 1  # set as 1 to avoid precision problems
# vacuum permittivity in [F/m]
epsilonZero = 8.854187817 * 1e-12 * 1e-2  # *(1e-2)  to make F/cm from F/m
# relative electric permittivity of the substrate: 3.9 SIO2
epsilon = 3.9
# oxidant thickness in [m]
tox = 285 * 1e-9 * 1e2  # *1e2 to make [cm]
#tox = 285
# capacitance of the oxidant on the gate, per unit of surface area
cox = epsilon * epsilonZero / tox  # divided by 1e4 to convert to square centimeters, 1e3 from epsilon & epsilonZero
# w f/cm^2
#prad w A
# mi CM2/Vm
#no w cm-2
#do 30 v sporobowac dopasowac
# poszukac chi2 minimization in scipy
#############

# Prepare logger
LOGGER = logging.getLogger('FITTER')
LOGGER.setLevel(logging.INFO)

handler = logging.FileHandler('fitter.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

# Reads data from a file and cuts it off at a specific place


def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        cutOffIndex = 0
        # cut off data not used for fitting
        for index in range(len(voltages)):
            if voltages[index] > 30:
                cutOffIndex = index
                break
        # choose a subset of data
        voltages = numpy.array(data[:cutOffIndex, 0])
        currents = numpy.array(data[:cutOffIndex, 1])
        currentsErr = numpy.array(data[:cutOffIndex, 2])

        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
        # to make kiloOhms
        scaledResistance = resistance/1e5
        return voltages, currents, currentsErr, resistance


def residual(voltages, mobility, rcontact, n0, vdirac):

    theory = 2 * rcontact + (sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility))  # to make kiloOhms
    return theory


def plotFigures(initialParameters, fileName, resultPath):

    resultName = os.path.split(os.path.splitext(fileName)[0])[1]
    voltages, currents, currentsErr, resistance = readData(fileName)
    resistance1 = numpy.array([a / b for a, b in zip(voltages, currents)])
    resistance2 = numpy.array([a / b for a, b in zip(resistance1, currents)])
    resitanceError = numpy.array([a * b for a, b in zip(resistance2, currentsErr)])
    # fitting to data using leastsq method

    best_parameters, covariance = curve_fit(residual, voltages, resistance, sigma=resitanceError,
                                             p0=initialParameters)
    print("parameters:", best_parameters)
    print("covariance:", covariance)
    # check if directory exists, create if needed
    directory = os.path.dirname(resultPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saving fit result to a text file
    #with open(os.path.join(resultPath, resultName+'_Fit.txt'), "w+") as fitResult:
    #    fitResult.write(best_parameters)
    #    fitResult.write(convariance)
    pyplot.figure()
    scatter = pyplot.scatter(voltages, resistance)
    initFitLine, = pyplot.plot(voltages, residual(voltages, initialParameters[0], initialParameters[1], initialParameters[2], initialParameters[3]), 'k--')
    bestFitLine, = pyplot.plot(voltages, residual(voltages, best_parameters[0], best_parameters[1], best_parameters[2], best_parameters[3]), 'r-')
    pyplot.xlabel('Gate voltage [ V ]')
    pyplot.ylabel('Resistance [ M\u2126 ]')
    pyplot.text(-8, 167, 'Sample dimension [length/width]: '+str(sampleDimension))
    pyplot.text(-8, 152, 'V_DS: 10 V')
    pyplot.title('Charakterystyka przejsciowa')
    pyplot.legend([scatter, bestFitLine, initFitLine], ['Data', 'Best Fit', 'Init fit'], loc='upper left')
    pyplot.savefig(os.path.join(resultPath, resultName))


# set initial parameters - mobility, rcontact, n0, vdirac

initialParameters = numpy.array([3e3, 1e5, 1e12, 60])

# system promts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: data directory/results): ") or filePath + '/results/'
sampleDimension = int(input("Write the sample dimensions (length/width, default: 6): ") or 6)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    plotFigures(initialParameters, infile, resultPath)

LOGGER.info('Script finished')
