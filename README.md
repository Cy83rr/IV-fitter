# IV-fitter
Python scripts used for fitting models to IV data
Opis warunków, w których było testowane oprogramowanie
**Data format** 
Entry data is organised in two or three columns, each separated with a tabulator. As a decimal point "," is used. The first row contains names for the rows. It is possible to change those assumptions in _readData_ function. The columns are respectively: voltage, current and current measurement error.
The script then outputs several files for each data file: one chart with plotted data, initial and best fit, one text file with fitting information and up to 6 correlation charts. Graphic files are saved in PNG format.
For convenience sake there is also a log file created.
**Installation:** 
Script is written using python3. Libraries used other than standard: LMFIT (high-level interface fofr scipy.optimize, used for data analysis), NumPy (data manipulation), Matplotlib (for data visualisation), Pandas (for useful data structures).
All of the packages can (and should be) installed thought pip3.
**Features:** 
Fitting a specific model to provided data, creating charts, computing linear correlations between fitting parameters and (in progress) creating correlation charts
**Usage:**
While running the script, there will be 4 prompts: one will be for data files path, one for script results path, one for sample dimensions (there is a default value assumed) and whether or not plotcorrelation charts. All default options will be listed, and it is possible to hit the enter should a user be satisfied with them.
**Acknowledgements for used libraries:** 
LMFIT (https://lmfit.github.io/lmfit-py/index.html) - _Newville, Matthew et al.. (2014). LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python¶. Zenodo. 10.5281/zenodo.11813_
From SciPy stack (https://www.scipy.org/about.html) :
NumPy & SciPy - Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37
Matplotlib - John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55
Pandas - Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)