import glob
import logging
import os
import sys

import lmfit
import matplotlib.pyplot as pyplot
import numpy
import pandas

# TODO: refactor and clean code

#############
# Constants
#############


# sample length/width, unitless - default value
sampleDimension = 6
# default value
ds_voltage = 0.01
# electron charge in [C]
echarge = 1.6021766208 * 1e-19
# vacuum permittivity in [F/m]
epsilonZero = 8.854187817 * 1e-12 * 1e-2  # *(1e-2)  to make F/cm from F/m
# relative electric permittivity of the substrate: 3.9 SIO2
epsilon = 3.9
# oxidant thickness in [m]
tox = 285 * 1e-9 * 1e2  # *1e2 to make [cm]
# capacitance of the oxidant on the gate, per unit of surface area
cox = 7.2*1e10*echarge
#############

# Prepare logger
LOGGER = logging.getLogger('FITTER')
LOGGER.propagate = True

handler = logging.FileHandler('fitter.log')
handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

LOGGER.addHandler(handler)
LOGGER.addHandler(console_handler)


def read_data(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        currents = numpy.array(data[:, 1])
        currents_err = numpy.array(data[:, 2])
        resistance = numpy.array(ds_voltage / currents)
        resistance_err = numpy.array(ds_voltage / (currents ** 2) * currents_err)
        return voltages, currents, resistance_err, resistance


def model(voltages, Rc, n0, vdirac, mobility):
    return 2 * Rc + \
        (sampleDimension /
            (numpy.sqrt(n0**2 + (cox * (voltages - vdirac) / echarge)**2) * echarge * mobility))


def plot_figures(initial_parameters, filename, result_path):

    result_name = os.path.split(os.path.splitext(filename)[0])[1]
    voltages, currents, resistance_err, resistance = read_data(filename)

    # fitting to data using leastsq method
    gmod = lmfit.Model(model)
    # TODO: why including the weights gives worse results?
    # Methods: by default, leastsq - Levenberg-Marquardt algorithm,
    # Many of the fit statistics and estimates for uncertainties are done for this only
    # Alternative methods: nelder, lbfgsb, powell, cg, newton, dogleg - more in lmfit docs
    result = gmod.fit(resistance, voltages=voltages, params=initial_parameters, method='leastsq')
    if not result.errorbars:
        LOGGER.log(logging.ERROR, msg='No error estimates in file: '+result_name)
        LOGGER.log(logging.ERROR, msg='Probable cause is too distant initial guess')
        LOGGER.log(logging.ERROR, msg=str(result.params))
    for key in initial_parameters.valuesdict():
        param = result.params.get(key)
        if param.stderr/param.value > 0.5:
            LOGGER.log(logging.ERROR, msg='Parameter ' + param.name + ' has over 50% error')
            LOGGER.log(logging.ERROR, msg='In file : ' + result_name)
    # check if directory exists, create if needed
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saving fit result to a text file
    with open(os.path.join(result_path, result_name + '_Fit.txt'), "w+") as fitResult:
        fitResult.write(result.fit_report())
    fits_path=os.path.join(result_path, 'fits.txt')
    if not os.path.exists(fits_path):
        with open(fits_path, "w+") as all_fits:
            all_fits.write('sample_name Rc n0 vdirac mobility')
    with open(fits_path, "w+") as all_fits:
        parameter_values = result.best_values
        line = "\n{} {} {} {} {}".format(result_name,parameter_values.get('Rc'), parameter_values.get('n0'),
                                         parameter_values.get('vdirac'), parameter_values.get('mobility'))
        all_fits.write(line)
    pyplot.figure()
    scatter = pyplot.scatter(voltages, resistance)
    best_fit_line, = pyplot.plot(voltages, result.best_fit, 'r-')
    pyplot.xlabel('Napięcie bramki [ V ]')
    pyplot.ylabel('Opór [ \u2126 ]')
    pyplot.figtext(0.15, 0.68, 'Napięcie dren-źródło: ' + str(ds_voltage) + 'V')
    pyplot.figtext(0.15, 0.65, 'Wymiar próbki: ' + str(sampleDimension))
    pyplot.title('Charakterystyka przejściowa')
    pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    pyplot.savefig(os.path.join(result_path, result_name))



# set initial parameters with bounds
init_parameters = lmfit.Parameters()
init_parameters.add('mobility', value=4e3, min=10) # in cm^2/V*s
init_parameters.add('Rc', value=1e3, min=10)  # in ohm
init_parameters.add('n0', value=1e12, min=1e10)  # in cm^-2
init_parameters.add('vdirac', value=50, min=0)  # in volts

# system prompts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: data directory/results): ") or filePath + '/results/'
sampleDimension = int(input("Sample dimension (length/width, default: 6): ") or 6)
ds_voltage = float(input("Drain/Source Voltage (in Volts, default: 0.01): ") or 0.01)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    plot_figures(init_parameters, infile, resultPath)

LOGGER.info('Script finished')
