import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

voltages = numpy.array([0])
currents = numpy.array([0])
resistance = numpy.array([0])

currentsErr= numpy.array([0])
rescale = 1.0


#todo: change starting parameters
#todo: doesnt modify starting parameters

def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        global voltages
        global currents
        global currentsErr
        global rescale
        global resistance
        voltages = numpy.array(data[:,0])
        currents = numpy.array(data[:,1])/100
        currentsErr = numpy.array(data[:,2])

        #resistance = abs(voltages/currents)
        resistance = numpy.array([a / b for a,b in zip(abs(voltages), currents)])
        #rescale = resistance.max()
        #resistance = resistance/rescale

def model(rcontact, n0, vdirac, mobility, voltages):
    #sample lenght/width, unitless
    sampleDimension=2
    #przenikalnosc elektryczna prozni
    epsilonZero=8.854187817*10**-12
    #wzgledna przenikalnosc elektyczna podloza -11.68 SI, 3.9 SIO2
    epsilon=3.9
    #pojemnosc elektryczna tlenku na bramce na jednostke powierzchni
    cox=285*10**-9/epsilon*epsilonZero
    #eletron charge
    echarge=1.6021766208*(10**-19)

    return 2 * rcontact + \
           (sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility))

def chisqfunction(params, voltagesData, resistanceData, currentsErr):
    rcontact = params['rcontact'].value
    n0 = params['n0'].value
    mobility = params['mobility'].value
    vdirac = params['vdirac'].value


    #niepewnosc oporu
    resitanceErr=abs((voltages/(currents**2))*currentsErr)

    predicted = model(rcontact, n0, vdirac, mobility, voltagesData)

    #chisq = numpy.sum(((resistance - predicted)/resitanceErr)**2)

    return (resistance - predicted)/resitanceErr

#initialParameters = numpy.array([1E9, 1e13, 60, 4000])
initialParameters = lmfit.Parameters()
initialParameters.add('mobility', value=4000)
initialParameters.add('rcontact', value=1E9)
initialParameters.add('n0', value=1E11)
initialParameters.add('vdirac', value=60)

readData('testData.txt')
result = lmfit.minimize(chisqfunction, initialParameters, args=(voltages, resistance, currentsErr))
print(result)

pyplot.scatter(voltages, resistance*rescale)
fittedParameters = result.params
rcontact = fittedParameters['rcontact'].value
n0 = fittedParameters['n0'].value
vdirac = fittedParameters['vdirac'].value
mobility = fittedParameters['mobility'].value
fittedData = resistance + result.residual
print(rcontact, n0, vdirac, mobility)
pyplot.plot(voltages, fittedData)
pyplot.savefig('plot2.png')