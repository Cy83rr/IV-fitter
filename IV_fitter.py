import matplotlib.pyplot as pyplot
import numpy
import pandas
from scipy.optimize import curve_fit

#############
# Constants
#############

# sample length/width, unitless
sampleDimension = 6
# electron charge in [C]
echarge = 1.6021766208 * 1e-19
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

# Reads data from a file and cuts it off at a specific place
resistance = numpy.array([])
voltages = numpy.array([])
resistance_error = numpy.array([])
currents = numpy.array([])
rescale = 1

with open("testData.txt") as dataFile:
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
    resistance1 = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
    resistance2 = numpy.array([a / b for a, b in zip(resistance1, currents)])
    resistance_error = numpy.array([a * b for a, b in zip(resistance2, currentsErr)])

    resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
    rescale = resistance.max()

    resistance = resistance / rescale
    resistance_error = resistance_error / rescale


def modelFunction(voltages, mobility, rcontact, n0, vdirac):
    model = 2 * rcontact + (sampleDimension / (numpy.sqrt(n0 ** 2 + (cox * (voltages - vdirac) / echarge) ** 2) * echarge * mobility))
    return model/rescale

def chisqFuction(initial_parameters):
    theory = modelFunction(voltages, initial_parameters)
    chisq=numpy.sum(((resistance - theory) / resistance_error) ** 2)
    return chisq

# set initial parameters - mobility, rcontact, n0, vdirac
initial_parameters = numpy.array([3e3, 1e5, 1e12, 60])

# set bounds for fitting parameters
bnds = ([1e2, 1e3, 1e8, 50], [2e5, 1e10, 1e16, 70])

# fitting to data using leastsq method, using minimum tolerance and displaying
result = curve_fit(modelFunction, voltages, resistance, p0=initial_parameters, bounds=bnds, sigma=resistance_error, method='dogbox')
best_parameters = result[0]
print("initial parameters:", initial_parameters)
print("parameters:", best_parameters)
print("errors: ", numpy.sqrt(numpy.diag(result[1])))
pyplot.figure()
scatter = pyplot.scatter(voltages, resistance*rescale)
initFitLine, = pyplot.plot(voltages, modelFunction(voltages, initial_parameters[0], initial_parameters[1], initial_parameters[2], initial_parameters[3]), 'k--')
bestFitLine, = pyplot.plot(voltages, modelFunction(voltages, best_parameters[0], best_parameters[1], best_parameters[2], best_parameters[3]), 'r-')
pyplot.xlabel('Gate voltage [ V ]')
pyplot.ylabel('Resistance [ M\u2126 ]')
pyplot.legend([scatter, bestFitLine, initFitLine], ['Data', 'Best Fit', 'Init fit'], loc='upper left')
pyplot.savefig("testData")







