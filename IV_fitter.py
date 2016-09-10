import glob
import logging
import os

import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: including current error - weighted least squares?

# TODO write all necessary dependencies or smth like gradle in java -
# TODO http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html

#############
# Constants
#############

# Flag for correlation charts
correlationCharts = False

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
        #for index in range(len(voltages)):
        #    if voltages[index] > 0:
        #        cutOffIndex = index
        #        break
        # choose a subset of data
        voltages = numpy.array(data[cutOffIndex:, 0])
        currents = numpy.array(data[cutOffIndex:, 1])
        currentsErr = numpy.array(data[cutOffIndex:, 2])

        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
        # to make kiloOhms
        scaledResistance = resistance/1e5
        return voltages, currents, currentsErr, resistance


def model(voltages, rcontact, n0, vdirac, mobility):
    return 2 * rcontact + (sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)) * echarge * mobility))  # to make kiloOhms


def plotCorrelationChart(trace, firstParameter, secondParameter, resultPath, resultName):
    #x, y, prob = trace[firstParameter][firstParameter], trace[firstParameter][secondParameter], trace[firstParameter]['prob']
    #x2, y2, prob2 = trace[secondParameter][secondParameter], trace[secondParameter][firstParameter], trace[secondParameter]['prob']
    #pyplot.figure()
    #pyplot.scatter(x, y, c=prob, s=30)
    #pyplot.scatter(x2, y2, c=prob2, s=30)
    #pyplot.xlabel(firstParameter)
    #pyplot.ylabel(secondParameter)

    x1, y1, prob1 = trace[firstParameter][firstParameter], trace[firstParameter][secondParameter], \
                    trace[firstParameter]['prob']
    y2, x2, prob2 = trace[secondParameter][secondParameter], trace[secondParameter][firstParameter], \
                    trace[secondParameter]['prob']

    pyplot.figure()
    pyplot.scatter(x1, y1, c=prob1, s=30)
    pyplot.scatter(x2, y2, c=prob2, s=30)
    ax = pyplot.gca()  # please label your axes!
    ax.set_xlabel(firstParameter)
    ax.set_ylabel(secondParameter)
    pyplot.savefig(os.path.join(resultPath, resultName+'_correlation_'+firstParameter+'_'+secondParameter))

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, prob1)
    ax.scatter(x2, y2, prob2)
    ax.set_xlabel(firstParameter)
    ax.set_ylabel(secondParameter)
    ax.set_zlabel('sigma')
    pyplot.savefig(os.path.join(resultPath, resultName+'_correlation3D_'+firstParameter+'_'+secondParameter))


def residual(params, voltages, resistance, resistanceError):
    n0 = params['n0']
    vdirac = params['vdirac']
    mobility = params['mobility']
    rcontact = params['rcontact']

    theory1 = float(2 * rcontact)
    theory2 = float(sampleDimension)
    theory3= float(n0**2)
    theory4= float(cox / echarge)
    theory5= voltages - vdirac

    theory6= numpy.sqrt(theory3 + theory4*theory5)
    theory7= echarge * mobility
    theory = theory1 + theory2/theory6/theory7

    return (resistance-theory)/resistanceError


def plotFigures(initialParameters, fileName, resultPath):

    resultName = os.path.split(os.path.splitext(fileName)[0])[1]
    voltages, currents, currentsErr, resistance = readData(fileName)
    resistance1 = numpy.array([a / b for a, b in zip(voltages, currents)])
    resistance2 = numpy.array([a / b for a, b in zip(resistance1, currents)])
    resitanceError = numpy.array([a * b for a, b in zip(resistance2, currentsErr)])
    # fitting to data using leastsq method
    #gmod = lmfit.Model(model)
    #result = gmod.fit(resistance, voltages=voltages, params=initialParameters)
    result = lmfit.minimize(residual, initialParameters, args=(voltages, resistance, resitanceError))
    if result.chisqr > 100:
        LOGGER.log(logging.ERROR, msg='Too big error in file: '+resultName)
        LOGGER.log(logging.ERROR, msg='Chisq: '+str(result.chisqr))
        LOGGER.log(logging.ERROR, msg=str(result.params))

    # check if directory exists, create if needed
    directory = os.path.dirname(resultPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saving fit result to a text file
    with open(os.path.join(resultPath, resultName+'_Fit.txt'), "w+") as fitResult:
        fitResult.write(result.fit_report())
    pyplot.figure()
    scatter = pyplot.scatter(voltages, resistance)
    initFitLine, = pyplot.plot(voltages, result.init_fit, 'k--')
    bestFitLine, = pyplot.plot(voltages, result.best_fit, 'r-')
    pyplot.xlabel('Gate voltage [ V ]')
    pyplot.ylabel('Resistance [ M\u2126 ]')
    pyplot.text(-8, 167, 'Sample dimension [length/width]: '+str(sampleDimension))
    pyplot.text(-8, 152, 'V_DS: 10 V')
    pyplot.title('Charakterystyka przejsciowa')
    pyplot.legend([scatter, initFitLine, bestFitLine], ['Data', 'Initial Fit', 'Best Fit'], loc='upper left')
    pyplot.savefig(os.path.join(resultPath, resultName))
    # TODO fix charts
    # Plot correlation charts
    if correlationCharts:
        LOGGER.info("Creating correlation charts")
        try:
            ci, trace = result.conf_interval(sigmas=[0.68, 0.95], trace=True, verbose=False)

            plotCorrelationChart(trace, 'n0', 'vdirac', resultPath, resultName)
            plotCorrelationChart(trace, 'mobility', 'n0', resultPath, resultName)
            plotCorrelationChart(trace, 'mobility', 'rcontact', resultPath, resultName)
            plotCorrelationChart(trace, 'mobility', 'vdirac', resultPath, resultName)
            plotCorrelationChart(trace, 'rcontact', 'n0', resultPath, resultName)
            plotCorrelationChart(trace, 'rcontact', 'vdirac', resultPath, resultName)
        except:
            LOGGER.error("Could not create charts for "+resultName)
        LOGGER.info("Finished creating correlation charts")


# set initial parameters with bounds
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=4e3, min=10, max=3*1e4)
initialParameters.add('rcontact', value=3e3, min=1e1, max=1e8)  # value times 1e5
initialParameters.add('n0', value=1e9, min=1e8, max=1e12)  # value times 1e6
initialParameters.add('vdirac', value=58, min=55, max=65)  # in volts

# system promts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: data directory/results): ") or filePath + '/results/'
sampleDimension = int(input("Write the sample dimensions (length/width, default: 6): ") or 6)
correlationCharts = bool(input("Plot correlation charts? (True/False, default: False): ") or False)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    plotFigures(initialParameters, infile, resultPath)

LOGGER.info('Script finished')
