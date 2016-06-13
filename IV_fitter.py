import glob
import os

import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: remember to properly cite lmfit library
# TODO: including current error - weighted least squares?
# TODO: write doc with necessary libraries to run program
# TODO: correlation charts
# TODO: make it a script
# TODO: add labels to chart
# TODO: remember to cite properly matplotlib and other libs!

# TODO write all necessary dependencies or smth like gradle in java -
# TODO http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html

#############
# Constants
#############

# sample length/width, unitless
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
        scaledResistance = resistance/1e6
        return voltages, currents, currentsErr, scaledResistance
# TODO check units!
def model(voltages, rcontact, n0, vdirac, mobility):


    # 1e4 and 1e-4 is for converting to square centimeters --> IS IT? CHECK, UNDERSTAND
    return 2 * rcontact + \
        (sampleDimension /
            (numpy.sqrt((n0*1e10*1e4)**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility*1e2*1e-4))\
        / 1e6  # to make kiloOhms

def chisqfunction(params, voltagesData, resistanceData, currentsData, currentsErr):
    rcontact = params['rcontact'].value
    n0 = params['n0'].value
    mobility = params['mobility'].value
    vdirac = params['vdirac'].value


    #  TODO:how to include those errors?
    #  resistance error
    #  global resitanceErr
    resitanceErr = numpy.array(abs((voltagesData/(currentsData**2))*currentsErr))

    predicted = numpy.array(model(voltagesData, rcontact, n0, vdirac, mobility))
    return numpy.array((predicted - resistanceData) / numpy.sqrt(predicted))

def plotFigures(initialParameters, fileName, resultPath):

    if not resultPath.strip():
        resultPath = os.path.curdir + '/results/'

    resultName = os.path.split(os.path.splitext(fileName)[0])[1]
    voltages, currents, currentsErr, resistance = readData(fileName)

    # fitting to data using lestsq method
    #gmod = lmfit.Model(model)
    #result = gmod.fit(resistance, voltages=voltages, params=initialParameters)

    minimizer = lmfit.Minimizer(chisqfunction, initialParameters,
                                fcn_args=(voltages, resistance, currents, currentsErr))
    result = minimizer.leastsq()
    # check if directory exists, create if needed
    directory = os.path.dirname(resultPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saving fit result to a text file
    with open(os.path.join(resultPath, resultName+'Fit.txt'), "w+") as fitResult:
        fitResult.write(lmfit.fit_report(result))

    # TODO: change the order of commands to draw - avoid needless redrawing
    #ci, trace = lmfit.conf_interval(minimizer, result, sigmas=[0.68, 0.95], trace=True, verbose=False)
    #x, y, prob = trace['mobility']['mobility'], trace['mobility']['vdirac'], trace['mobility']['prob']
    #x2, y2, prob2 = trace['vdirac']['vdirac'], trace['vdirac']['mobility'], trace['vdirac']['prob']
    #fig1 = pyplot.figure(0)
    # ax1 = fig1.add_subplot(111)
    #pyplot.scatter(x, y, c=prob)
    #pyplot.scatter(x2, y2, c=prob2)
    #pyplot.xlabel('mobility')
    #pyplot.ylabel('vdirac')
    #pyplot.savefig(os.path.join(resultPath, resultName+'_mob-n0.png'))

    print(lmfit.fit_report(result, min_correl=0.1))
    fittedParameters = result.params
    fittedData = model(voltages, fittedParameters.get('rcontact').value,
                       fittedParameters.get('n0').value,
                       fittedParameters.get('vdirac').value,
                       fittedParameters.get('rcontact').value)
    fig2 = pyplot.figure(1)
    # ax2 = fig2.add_subplot(111)
    scatter = pyplot.scatter(voltages, resistance)
    line, = pyplot.plot(voltages, fittedData, color='red')
    pyplot.xlabel('Gate voltage [ V ]')
    pyplot.ylabel('Resistance [ k\u2126 ]')
    pyplot.text(3, 60, 'Sample dimension [length/width]: '+str(sampleDimension))
    pyplot.text(3, 55, 'V_DS: 10 V')
    # ax2.errorbar(voltages, resistance, resistanceErr)
    pyplot.legend((scatter, line), ['Data', 'Fit'], loc="best")
    # TODO: save additional information on the chart: sample dimensions, source-drain voltage
    pyplot.savefig(os.path.join(resultPath, resultName))

# set initial parameters with bounds
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=20, min=1, max=2e2)  # value times 1e2
initialParameters.add('rcontact', value=0.01, min=0, max=1e1)  # value times 1e6
initialParameters.add('n0', value=10, min=0.1, max=1e3)  # value times 1e10
initialParameters.add('vdirac', value=60, min=55, max=65)

# system promts for data input/output
filePath = input("Write the path to data files (default: current directory): ")
resultPath = input("Write the path where the results will be saved(default: data directory/results): ")
sampleDimension = int(input("Write the sample dimensions (length/width, default: 2): "))

for infile in glob.glob( os.path.join(filePath, '*.txt')):
    plotFigures(initialParameters, infile, resultPath)

