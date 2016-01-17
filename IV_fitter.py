import matplotlib.pyplot as pyplot
import numpy
import pandas
import scipy.optimize as optimize


voltages = []
currents = []
currentsErr= []
rescale = 1.0
#sample lenght/width, unitless
sampleDimension=80
#pojemnosc elektryczna tlenku na bramce na jednostke powierzchni
cox=50
#eletron charge
echarge=1.6*10**-10
#gate voltage in V
vgate=10

def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',', skiprows=1).values
        global voltages
        global currents
        global currentsErr
        global rescale
        voltages = data[:,0]
        currents = data[:,1]
        currentsErr = data[:,2]/1000
        rescale = currents.max()

        print(voltages)
        print(currents)
        print(currentsErr)
        voltages = voltages/rescale

def chisqfunction(rcontact_n0_vdirac_mobility):
    rcontact,n0,vdirac,mobility = rcontact_n0_vdirac_mobility


    #niepewnosc oporu
    resitanceErr=voltages/(currents**2)*currentsErr

    #todo: extract model function for readibility
    model = 2*rcontact+(sampleDimension/ (numpy.sqrt( (n0+cox*(voltages-vdirac)/echarge)**2) *echarge*mobility ))

    chisq = numpy.sum(((voltages/currents- model)/resitanceErr)**2)

    return chisq

x0 = numpy.array([0,0,0,0])

readData('testData.txt')
result =  optimize.minimize(chisqfunction, x0)
print(result)
assert result.success==True

pyplot.scatter(currents, voltages*rescale)
rcontact,n0,vdirac,mobility=result.x*rescale
print(rcontact,n0,vdirac,mobility)
pyplot.plot(2*rcontact+(sampleDimension/ (numpy.sqrt( (n0+cox*(voltages-vdirac)/echarge)**2) *echarge*mobility )), voltages )
pyplot.savefig('firstplot.png')