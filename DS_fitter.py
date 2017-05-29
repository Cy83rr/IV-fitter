import glob
import logging
import os

import matplotlib.pyplot as pyplot
import numpy
import pandas

#############
# Constants
#############


# sample length/width, unitless - default value
sampleDimension = 6
# default value
gate_voltage = 0.01
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
LOGGER.setLevel(logging.INFO)

handler = logging.FileHandler('fitter.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

LOGGER.addHandler(handler)

# TODO different file structure, multiple series such as: U I dI I dI etc. gate voltage is in current column name
# TODO use lists for currents and errors, iterate
def read_data(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\t', decimal=',').values
        voltages = numpy.array(data[:, 0])
        number_of_currents = len(data[1])-1
        currents = []
        currents_err = []
        for data_index in range(1, number_of_currents - 1, 2):
            currents.append(numpy.array(data[:, data_index]))
            currents_err.append(numpy.array(data[:, data_index+1]))
        resistance = numpy.array(gate_voltage / numpy.array(currents))
        resistance_err = numpy.array(gate_voltage / (numpy.array(currents) ** 2) * numpy.array(currents_err))
        return voltages, resistance_err, resistance


def plot_figures(data, result_path, result_name):

    voltages, resistance_err, resistance = data

    # check if directory exists, create if needed
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    number_of_currents = len(resistance)

    pyplot.figure()
    for resistance_data in resistance:
        pyplot.plot(voltages, resistance_data)
    pyplot.xlabel('Napięcie dren-źródło [ V ]')
    pyplot.ylabel('Opór [ \u2126 ]')
    pyplot.figtext(0.15, 0.68, 'Napięcie bramki: ' + str(gate_voltage) + 'V')
    pyplot.figtext(0.15, 0.65, 'Wymiar próbki: ' + str(sampleDimension))
    pyplot.title('Charakterystyka wyjśćiowa')
    # legenda
    #pyplot.legend([scatter, ], ['Dane', ''], loc='upper left')
    pyplot.savefig(os.path.join(result_path, result_name))

# system prompts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: current directory/results): ") or os.path.curdir + '/results/'
sampleDimension = int(input("Sample dimension (length/width, default: 6): ") or 6)
gate_voltage = float(input("Gate Voltage (in Volts, default: 0.01): ") or 0.01)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    data = read_data(infile)
    result_name = os.path.split(os.path.splitext(infile)[0])[1]
    plot_figures(data, resultPath, result_name)

LOGGER.info('Script finished')
