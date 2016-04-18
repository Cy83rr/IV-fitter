import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: remember to properly cite lmfit library
# TODO: including current error - weighted least squares?
# TODO: write doc with neccessary libraries to run program
# TODO: correlation charts
# TODO: fit parameters have huge errors - precision problem?
# TODO: make it a script
# TODO: decide, how it should look like user-wise
# TODO: add labels to chart

# TODO write all necessary dependencies or smth like gradle in java -
# TODO http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html

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

        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])  # to make kiloOhms
        return voltages, currents, currentsErr, resistance

def model(rcontact, n0, vdirac, mobility, voltages):
    # sample length/width, unitless
    sampleDimension=2
    # vaccum permittivity
    epsilonZero=8.854187817*1e-12
    # relative electric permittivity of the substrate: 11.68 SI, 3.9 SIO2
    epsilon=3.9
    # capacitance of the oxydant on the gate, per unit of surface area
    cox=epsilon*epsilonZero/(285*1e-9) / 1e4 # divided by 1e4 to convert to square centimeters
    # eletron charge
    echarge=1.6021766208*(1e-19)

    return 2 * rcontact*1e7 + \
            (sampleDimension / (numpy.sqrt((n0*1e9)**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility*1e3))

def chisqfunction(params, voltagesData, resistanceData, currentsData, currentsErr):
    rcontact = params['rcontact'].value
    n0 = params['n0'].value
    mobility = params['mobility'].value
    vdirac = params['vdirac'].value


    # TODO: how to include those errors?
    # resistance error
    resitanceErr = numpy.array(abs((voltagesData/(currentsData**2))*currentsErr))

    predicted = numpy.array(model(rcontact, n0, vdirac, mobility, voltagesData))
    return numpy.array(((resistanceData - predicted)/predicted))
# set initial parameters with bounds
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=1, min=1e-1, max=100)
initialParameters.add('rcontact', value=3, min=1e-4)
initialParameters.add('n0', value=1.2, min=1e-3)
initialParameters.add('vdirac', value=60, min=58, max=65)

voltages, currents, currentsErr, resistance = readData('testData.txt')
# TODO: use a minimizer, might help counting stderr and correl and is (?) needed for correl charts
minimizer = lmfit.Minimizer(chisqfunction, initialParameters, fcn_args=(voltages, resistance, currents, currentsErr))
result = minimizer.leastsq()

print(result)

#ci, trace = lmfit.conf_interval(minimizer, result, sigmas=[0.68,0.95], trace=True, verbose=False)
#pyplot.plot(trace)
#pyplot.savefig('trace.png')
#pyplot.new_figure_manager
pyplot.scatter(voltages, resistance)

fittedParameters = result.params

fittedData = resistance + result.residual

print(lmfit.fit_report(result, min_correl=0.1))
pyplot.plot(voltages, fittedData)
pyplot.savefig('plot2.png')