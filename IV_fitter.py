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

def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        global voltages
        global currents
        global currentsErr
        global rescale
        global resistance
        voltages = numpy.array(data[:,0])
        currents = numpy.array(data[:,1])
        currentsErr = numpy.array(data[:,2])

        #resistance = abs(voltages/currents)
        resistance = numpy.array([a / b for a,b in zip(abs(voltages), currents)])
        #rescale = resistance.max()
        #resistance = resistance/rescale
def model(rcontact, n0, vdirac, mobility, voltages):
    #sample lenght/width, unitless
    sampleDimension=80
    #pojemnosc elektryczna tlenku na bramce na jednostke powierzchni
    cox=50
    #eletron charge
    echarge=1.6*10**-10

    return 2*rcontact+(sampleDimension/ (numpy.sqrt( (n0+cox*(voltages-vdirac)/echarge)**2) *echarge*mobility ))

def chisqfunction(rcontact_n0_vdirac_mobility):
    rcontact,n0,vdirac,mobility = rcontact_n0_vdirac_mobility


    #niepewnosc oporu
    #todo: recount - expected type Number instead of list - voltages anad currents
    resitanceErr=abs((voltages/(currents**2))*currentsErr)

    #todo: extract model function for readibility
    #todo: substraction is invalid for voltages
    predicted = model(rcontact, n0, vdirac, mobility, voltages)

    chisq = numpy.sum(((resistance - predicted)/resitanceErr)**2)

    return chisq

x0 = numpy.array([1000000,4000,20,4000])

readData('testData.txt')
result = optimize.minimize(chisqfunction, x0)
print(result)
assert result.success==True

pyplot.scatter(voltages, resistance*rescale)
rcontact,n0,vdirac,mobility=result.x*rescale
print(rcontact,n0,vdirac,mobility)
pyplot.plot(voltages ,model(rcontact, n0, vdirac, mobility, voltages))
pyplot.savefig('plot2.png')