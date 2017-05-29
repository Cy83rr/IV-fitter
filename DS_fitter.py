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


def read_data(filename):
    with open(filename) as dataFile:
        columns=pandas.read_csv(dataFile, sep='\t', decimal=',')
        column_names=list(columns)
        gate_voltages_strings = []
        gate_voltages = []
        for current_index in range(1, len(column_names)-1, 2):
            gate_voltages_strings.append(column_names[current_index])
        for record in gate_voltages_strings:
            gate_voltages.append(float(''.join(c for c in record if c not in 'I()').replace(',', '.')))
        data = columns.values
        voltages = numpy.array(data[:, 0]) * 1e3  # w mV
        number_of_currents = len(data[1])
        currents = []
        currents_err = []
        # scale currents to microampers
        for data_index in range(1, number_of_currents - 1, 2):
            currents.append(numpy.array(data[:, data_index]) * 1e6)
            currents_err.append(numpy.array(data[:, data_index+1]) * 1e6)

        return voltages, numpy.array(currents_err), numpy.array(currents), gate_voltages


def plot_figures(data, result_path, result_name):

    voltages, currents_err, currents, gate_voltages = data

    # check if directory exists, create if needed
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pyplot.figure()
    for i in range(0, len(currents)):
        pyplot.errorbar(voltages, currents[i], currents_err[i])
    pyplot.xlabel('Napięcie dren-źródło [ mV ]')
    pyplot.ylabel('Natężenie dren-źródło [ μA ]')
    pyplot.figtext(0.15, 0.68, 'Napięcia bramki od: ' + str(gate_voltages[0]) + ' V do '
                   + str(gate_voltages[len(gate_voltages)-1]) + ' V ')
    pyplot.figtext(0.15, 0.65, 'Wymiar próbki: ' + str(sampleDimension))
    pyplot.title('Charakterystyka wyjściowa')
    pyplot.savefig(os.path.join(result_path, result_name))

# system prompts for data input/output
filePath = input("Write the path to data files (default: current directory): ") or os.path.curdir
resultPath = input("Write the path where the results will be saved(default: current directory/results): ") or os.path.curdir + '/results/'
sampleDimension = int(input("Sample dimension (length/width, default: 6): ") or 6)

LOGGER.info('Script starting')

# Iterate over every file with .txt extension
for infile in glob.glob(os.path.join(filePath, '*.txt')):
    data = read_data(infile)
    result_name = os.path.split(os.path.splitext(infile)[0])[1]
    plot_figures(data, resultPath, result_name)

LOGGER.info('Script finished')
