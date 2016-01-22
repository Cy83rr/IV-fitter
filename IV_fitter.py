import matplotlib.pyplot as pyplot
import numpy
import pandas
import scipy.optimize as optimize


voltages = numpy.array([0])
currents = numpy.array([0])
resistance = numpy.array([0])

currentsErr= numpy.array([0])
rescale = 1.0


#todo: divide list by another list problem
#todo: substract from list problem
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
    epsilon=11.68
    #pojemnosc elektryczna tlenku na bramce na jednostke powierzchni
    cox=285*10**-9/epsilon*epsilonZero
    #eletron charge
    echarge=1.6021766208*(10**-19)

    return 2*rcontact+(sampleDimension / (numpy.sqrt(n0**2 + (cox * (voltages-vdirac) / echarge)**2) * echarge * mobility))

def chisqfunction(rcontact_n0_vdirac_mobility):
    rcontact,n0,vdirac,mobility = rcontact_n0_vdirac_mobility


    #niepewnosc oporu
    #todo: recount - expected type Number instead of list - voltages anad currents
    resitanceErr=abs((voltages/(currents**2))*currentsErr)

    print(resitanceErr)
    #todo: substraction is invalid for voltages
    predicted = model(rcontact, n0, vdirac, mobility, voltages)
    print(predicted)

    chisq = numpy.sum(((resistance - predicted)/resitanceErr)**2)

    return chisq

initialParameters = numpy.array([1E9, 1E11, 60, 8000])

readData('testData.txt')
result = optimize.minimize(chisqfunction, initialParameters)
print(result)
assert result.success==True

pyplot.scatter(voltages, resistance*rescale)
rcontact, n0, vdirac, mobility = result.x
print(rcontact, n0, vdirac, mobility)
pyplot.plot(voltages, model(rcontact, n0, vdirac, mobility, voltages))
pyplot.savefig('plot2.png')