import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: remember to properly cite lmfit library
# TODO: including current error - weighted least squares
# TODO: write doc with neccessary libraries to run program
# TODO: correlation charts
# TODO: make it a script
# TODO: decide, how it should look like user-wise
# TODO: fit parameters have 0 standard error - why?
# TODO: add labels to chart

def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        cutOffIndex = 0
        for index in range(len(voltages)):
            if voltages[index] > 0:
                cutOffIndex = index
                break
        # choose a subset of data
        voltages = numpy.array(data[cutOffIndex:, 0])
        currents = numpy.array(data[cutOffIndex:, 1])
        currentsErr = numpy.array(data[cutOffIndex:, 2])

        resistance = numpy.array([a / b for a, b in zip(abs(voltages), currents)])
        return voltages, currents, currentsErr, resistance

def model(rcontact, n0, vdirac, mobility, voltages):
    # sample length/width, unitless
    sampleDimension=2
    # przenikalnosc elektryczna prozni
    epsilonZero=8.854187817*10**-12
    # wzgledna przenikalnosc elektyczna podloza: 11.68 SI, 3.9 SIO2
    epsilon=3.9
    # pojemnosc elektryczna tlenku na bramce na jednostke powierzchni
    cox=285*10**-9/epsilon*epsilonZero
    # eletron charge
    echarge=1.6021766208*(10**-19)

    return 2 * rcontact + \
            (sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility))

def chisqfunction(params, voltagesData, resistanceData, currentsData, currentsErr):
    rcontact = params['rcontact'].value
    n0 = params['n0'].value
    mobility = params['mobility'].value
    vdirac = params['vdirac'].value

    # TODO: how to include this errors?
    # niepewnosc oporu
    resitanceErr = abs((voltagesData/(currentsData**2))*currentsErr)

    predicted = model(rcontact, n0, vdirac, mobility, voltagesData)

    return (resistanceData - predicted)/predicted
# set initial parameters with bounds
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=4000, min=0, max=15000)
initialParameters.add('rcontact', value=1E6, min=0)
initialParameters.add('n0', value=1E9, min=0)
initialParameters.add('vdirac', value=60, min=55, max=65)

voltages, currents, currentsErr, resistance = readData('testData.txt')
# TODO: use a minimizer, might help counting stderr and correl and is (?) needed for correl charts
minimizer = lmfit.Minimizer(chisqfunction, initialParameters, fcn_args=(voltages, resistance, currents, currentsErr))
halfResult = minimizer.minimize(method='nelder-mead')
print(halfResult)
result = minimizer.minimize(method='leastsq', params=halfResult.params)
print(result)

ci, trace = lmfit.conf_interval(minimizer, result, sigmas=[0.68,0.95],
                                trace=True, verbose=False)
pyplot.plot(trace)
pyplot.savefig('trace.png')
pyplot.new_figure_manager
pyplot.scatter(voltages, resistance)

fittedParameters = result.params
rcontact = fittedParameters['rcontact'].value
n0 = fittedParameters['n0'].value
vdirac = fittedParameters['vdirac'].value
mobility = fittedParameters['mobility'].value

fittedData = resistance + result.residual

print(lmfit.fit_report(result))
pyplot.plot(voltages, fittedData)
pyplot.savefig('plot2.png')