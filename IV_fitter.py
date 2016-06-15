import glob
import logging
import os

import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: including current error - weighted least squares?
# TODO: write doc with necessary libraries to run program
# TODO: correlation charts
# TODO: remember to cite properly matplotlib and other libs!
# TODO: remember to properly cite lmfit library

# TODO: log when there are errors in correlation charts


# TODO write all necessary dependencies or smth like gradle in java -
# TODO http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html

#############
# Constants
#############

# sample length/width, unitless - default value
sampleDimension = 2
# electron charge in [C]
echarge = 1.6021766208 * 1e-19
# vaccum permittivity in [F/m]
epsilonZero = 8.854187817 * 1e-12  # *(1e-2)  to make F/cm from F/m
# relative electric permittivity of the substrate: 11.68 SI, 3.9 SIO2
epsilon = 3.9
# oxidant thickness in [m]
tox = 285 * 1e-9  # *1e2 to make [cm]
# capacitance of the oxidant on the gate, per unit of surface area
cox = epsilon * epsilonZero / tox  # divided by 1e4 to convert to square centimeters

#############

#Prepare logger
LOGGER = logging.getLogger('FITTER')
LOGGER.setLevel(logging.INFO)

handler = logging.FileHandler('fitter.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

# read data from a file and cuts it off at a specific place
def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        cutOffIndex = 0
        # cut off data not used for fitting
        for index in range(len(voltages)):
            if voltages[index] > 0:
                cutOffIndex = index
                break
        # choose a subset of data
        voltages = numpy.array(data[cutOffIndex:, 0])
        currents = numpy.array(data[cutOffIndex:, 1])
        currentsErr = numpy.array(data[cutOffIndex:, 2])

        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
        # to make kiloOhms
        scaledResistance = resistance/1e5
        return voltages, currents, currentsErr, scaledResistance
# TODO check units!
def model(voltages, rcontact, n0, vdirac, mobility):
    # 1e4 and 1e-4 is for converting to square centimeters --> IS IT? CHECK, UNDERSTAND
    return 2 * rcontact + \
        (sampleDimension /
            (numpy.sqrt((n0*1e9)**2 + (cox * 1e-4 * (voltages - vdirac) / echarge)**2) * echarge * mobility*1e2))\
        / 1e5  # to make kiloOhms

def plotFigures(initialParameters, fileName, resultPath):

    resultName = os.path.split(os.path.splitext(fileName)[0])[1]
    voltages, currents, currentsErr, resistance = readData(fileName)

    # fitting to data using leastsq method
    gmod = lmfit.Model(model)
    result = gmod.fit(resistance, voltages=voltages, params=initialParameters)
    if result.chisqr > 100:
        LOGGER.log(logging.ERROR, msg='Too big error in file: '+resultName)
        LOGGER.log(logging.ERROR, msg='Chisq: '+str(result.chisqr))
        LOGGER.log(logging.ERROR, msg=str(result.params))

    # check if directory exists, create if needed
    directory = os.path.dirname(resultPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saving fit result to a text file
    with open(os.path.join(resultPath, resultName+'Fit.txt'), "w+") as fitResult:
        fitResult.write(result.fit_report())
# TODO remove
    print(lmfit.fit_report(result, min_correl=0.1))
    pyplot.figure()
    scatter = pyplot.scatter(voltages, resistance)
    initFitLine = pyplot.plot(voltages, result.init_fit, 'k--')
    bestFitLine = pyplot.plot(voltages, result.best_fit, 'r-')
    pyplot.xlabel('Gate voltage [ V ]')
    pyplot.ylabel('Resistance [ M\u2126 ]')
    # TODO fix text placement
    pyplot.text(-8, 47, 'Sample dimension [length/width]: '+str(sampleDimension))
    pyplot.text(-8, 42, 'V_DS: 10 V')
    pyplot.title('Charakterystyka przejsciowa')
    pyplot.legend((scatter, initFitLine, bestFitLine), ['Data', 'Initial Fit', 'Best Fit'], loc='best')
    pyplot.savefig(os.path.join(resultPath, resultName))

    # TODO: correlation charts
    #pyplot.figure()


# set initial parameters with bounds
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=10, min=1, max=2e2)  # value times 1e2
initialParameters.add('rcontact', value=1, min=0.01, max=1e3)  # value times 1e5
initialParameters.add('n0', value=10, min=0, max=1e3)  # value times 1e9
initialParameters.add('vdirac', value=60, min=55, max=65) # in volts

# system promts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: data directory/results): ") or filePath + '/results/'
sampleDimension = int(input("Write the sample dimensions (length/width, default: 6): ") or 6)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    plotFigures(initialParameters, infile, resultPath)

LOGGER.info('Script finished')

