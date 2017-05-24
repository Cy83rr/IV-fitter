import logging
import os

import matplotlib.pyplot as pyplot
import numpy
import pandas
import scipy.stats as stats

#TODO lepszy logger
#TODO poprawic odpalanie skryptu

# Prepare logger
LOGGER = logging.getLogger('FITTER')
LOGGER.setLevel(logging.INFO)

handler = logging.FileHandler('fitter.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

LOGGER.addHandler(handler)


# read data from file
def read_data(filename):
    with open(filename) as dataFile:
        data = pandas.read_csv(dataFile, sep='\s+', decimal=',').values
        rc = numpy.array(data[:, 1])
        n0 = numpy.array(data[:, 2])
        vdirac = numpy.array(data[:, 3])
        mobility = numpy.array(data[:, 4])
        return rc, n0, vdirac, mobility


def count_correl(variables):
    rc, n0, vdirac, mobility = variables
    rc_n0 = stats.pearsonr(rc, n0)[0]
    rc_vdirac = stats.pearsonr(rc, vdirac)[0]
    rc_mobility = stats.pearsonr(rc, mobility)[0]
    vdirac_n0 = stats.pearsonr(vdirac, n0)[0]
    vdirac_mobility = stats.pearsonr(vdirac, mobility)[0]
    n0_mobility = stats.pearsonr(n0,mobility)[0]
    return rc_n0, rc_vdirac, rc_mobility, vdirac_n0, vdirac_mobility, n0_mobility

# TODO: jednostki na wykresach
def plot_figures(variables, correlations, result_path):
    rc, n0, vdirac, mobility = variables
    rc_n0, rc_vdirac, rc_mobility, vdirac_n0, vdirac_mobility, n0_mobility = correlations

    pyplot.figure()
    pyplot.scatter(rc, n0)
    pyplot.xlabel('Opór kontaktów')
    pyplot.ylabel('Koncentracja nośników')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(rc_n0))
    pyplot.title('Wykres rozrzutu')
    #pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name='Correlation_rc_n0'
    pyplot.savefig(os.path.join(result_path, figure_name))

    pyplot.figure()
    pyplot.scatter(rc, vdirac)
    pyplot.xlabel('Opór kontaktów')
    pyplot.ylabel('Napięcie Diraca')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(rc_vdirac))
    pyplot.title('Wykres rozrzutu')
    # pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name = 'Correlation_rc_vdirac'
    pyplot.savefig(os.path.join(result_path, figure_name))

    pyplot.figure()
    pyplot.scatter(rc, mobility)
    pyplot.xlabel('Opór kontaktów')
    pyplot.ylabel('Ruchliwość nośników')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(rc_mobility))
    pyplot.title('Wykres rozrzutu')
    # pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name = 'Correlation_rc_mobility'
    pyplot.savefig(os.path.join(result_path, figure_name))

    pyplot.figure()
    pyplot.scatter(vdirac, n0)
    pyplot.xlabel('Napięcie Diraca')
    pyplot.ylabel('Koncentracja nośników')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(vdirac_n0))
    pyplot.title('Wykres rozrzutu')
    # pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name = 'Correlation_vdirac_n0'
    pyplot.savefig(os.path.join(result_path, figure_name))

    pyplot.figure()
    pyplot.scatter(vdirac, mobility)
    pyplot.xlabel('Napięcie Diraca')
    pyplot.ylabel('Ruchliwość nośników')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(vdirac_mobility))
    pyplot.title('Wykres rozrzutu')
    # pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name = 'Correlation_vdirac_mobility'
    pyplot.savefig(os.path.join(result_path, figure_name))

    pyplot.figure()
    pyplot.scatter(n0, mobility)
    pyplot.xlabel('Koncentracja nośników')
    pyplot.ylabel('Ruchliwość nośników')
    pyplot.figtext(0.15, 0.68, 'Współczynnik korelacji Pearsona: ' + str(n0_mobility))
    pyplot.title('Wykres rozrzutu')
    # pyplot.legend([scatter, best_fit_line], ['Dane', 'Dopasowanie'], loc='upper left')
    figure_name = 'Correlation_n0_mobility'
    pyplot.savefig(os.path.join(result_path, figure_name))

filePath = input("Write the path and filename to corelation data (default: current directory): ") or os.path.curdir + '/results/fits.txt'
resultPath = input("Write the path where the results will be saved(default: current directory/results): ") or os.path.curdir + '/results/'
variables=read_data(filePath)
correlations = count_correl(variables)
plot_figures(variables, correlations, resultPath)



