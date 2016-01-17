import matplotlib.pyplot as pyplot
import numpy
import pandas
import scipy.optimize as optimize


voltages = []
currents = []
currentsErr= []
rescale = 1.0

def readData(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',', skiprows=1).values
        global voltages
        global currents
        global currentsErr
        global rescale
        print(data)
        voltages = data[:,0]
        currents = data[:,1]
        currentsErr = data[:,2]/1000
        rescale = currents.max()

        print(voltages)
        print(currents)
        print(currentsErr)
        voltages = voltages/rescale

def chisqfunction(a_b):
    a, b = a_b
    model = a + b*currents
    chisq = numpy.sum(((voltages- model)/currentsErr)**2)
    return chisq

x0 = numpy.array([0,0])

readData('testData.txt')
result =  optimize.minimize(chisqfunction, x0)
print(result)
assert result.success==True

pyplot.scatter(currents, voltages*rescale)
a,b=result.x*rescale
print(a,b)
pyplot.plot(currents, a+b*currents)
pyplot.savefig('firstplot.png')