# CO bandhead disk model

This code calculates parametrized LTE models of CO (overtone) bandhead emission, from a flat Keplarian disk with a radial temperature and column density structure. Detailed description of the model and its scientific motivation can be found in the paper "Massive pre-main-sequence stars in M17: First and second overtone CO bandhead emission and the thermal infrared." by [J. Poorta et al.](https://ui.adsabs.harvard.edu/abs/2023arXiv230501436P/abstract) (DOI: https://doi.org/10.1051/0004-6361/202245658). The exact version of the code used for the paper can be found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7774529.svg)](https://doi.org/10.5281/zenodo.7774529)


## Installation and dependencies
To run the code and calculate (grids) of models the repository can be downloaded. 

The exact environment (which was created under Python 3.7.5) used to develop the code can be found in `requirements.txt`. However, the essential packages are `numpy`, `matplotlib`, `pandas`, and `scipy`, and with these installed the code will likely work under any Python 3 distribution. 

All relevant modules are in the ```model``` directory.  To get started run ``example.py`` which calculates a model and produces a few plots.

## Documentation

For detailed documentation per module and method see the GitHub [pages](https://jpoorta.github.io/co_bandhead_disk_model/index.html) documentation of this project.

## B



The <green>**software**</green>  folder contains the full version of the code which was used to run the final grid that was used to fit and perform the error analysis as presented in the paper. No installation is necessary - one can simply download the folder and run scripts as described below. 

For the paper this code was run on the LISA Compute Cluster (https://www.surf.nl/en/lisa-compute-cluster-extra-processing-power-for-research) and for that purpose parallelized to run through the grid in 16 chunks (see ```grid.sh```). The environment under which the code was developed and tested, and also in which all the postprocessing and plotting was done can be found in <mag>**requirements.txt**</mag>. Python version 3.7.5. was used. However, the code will likely work fine under any Python 3 distribution and the main packages of importance are ```numpy``` and ```matplotlib```.

The <green>**software**</green> folder contains the following files and folders:
- <green>**flat_disk_model_repro**</green>
  > NOTE: To run scripts as described below, navigate into <green>**flat_disk_model_repro**</green> first.
  - ```test.py``` -- This script calculates and plots one model. After setting the path to where <green>**software**</green> is stored, as instructed in the file header, it can be run from the command  line as follows:
    ```bash
    python3 test.py
    ```
    It produces the four plots stored in <green>**test_plots**</green>.
  - ```grid.sh``` -- This is an example script for running the code in LISA. It exemplifies how ```co_bandhead_grid.py``` can be run on 16 parallel cores, by calling ```co_bandhead_grid.py``` 16 times with a different thread number.
  -  ```co_bandhead_grid.py``` -- This script can be run to reproduce a full model grid *for each object* (but see caveat under <green>**results**</green>). After setting the path to where <green>**software**</green> is stored, as instructed in the file header, this script can be run from the command  line as follows:
     ```bash
      python3 co_bandhead_grid.py <i>
     ```
     
      where ```i``` is the thread number, an integer that runs from 0 to 15. Thus, to reproduce the full grid for five objects, this script has to be run 16 times (see for example ```grid.sh```).  
    >   NOTE: The number of parallel cores (16) is hard coded. This can be changed in the function ```run_grid_log_r``` in ```flat_disk_log_grid.py```. For more information, see the docstring and usage of the parameter ```lisa_it``` of that function.  

      Alternatively, one can calculate the whole model grid for all objects serially by simply calling the script without input:
    ```bash
    python3 co_bandhead_grid.py
    ```
    This will put the thread number (the ```lisa_it``` parameter) to ```None```. 

    All output will be stored in the <green>**results**</green> folder (see below).
  - ```config.py``` This script declares all global constants and parameters, reads in the necessary data, sets folder paths, contains widely used general functions (e.g. black body function) etc. Most parameters and functions are explained in docstrings and/or comments. Note that this script also contains some redundant functions that are not used by any scripts.
  - ```flat_disk_log_grid.py``` The module containing all the actual model grid calculations. Again, more information can be found in the docstrings and in-code comments.
  - ```sed_calculations.py``` Contains a function to calculate an SED from dust-disk and stellar parameter input. 
  - <green>**aux_files**</green> -- Contains the necessary input- and data-files used in the model. The <or>**\*.pkl**</or> files are pickle files that contain, per object, the best fit information of their respective SEDs. The <or>**\*.npy**</or> files have to be read with using numpy. All other files are text files. Most (if not all) of the files in this folder are read by and explained in ```config.py```. 
  - <green>**test_plots**</green> -- Contains the four plots produced by ```test.py```.
  
  
- <green>**results**</green>  
  >     <red>CAVEAT: Not all models are saved!</red> (See explanation of <mag>**out.txt**</mag>).
 
  This is where the output of ```co_bandhead_grid.py``` will be stored (set in ```config.py```). The folder was added to this package as an illustration, and to have the necessary folder structure in place to run the grid as described above. It contains folders with object names, which contain a folder with the grid name, where grid models will be stored. The grid name is an input parameter which can be set in ```co_bandhead_grid.py```. Some example output is included in  <green>**B268**</green>:

  - <green>**B268/grid_21_5_2021_dv**</green> -- B268 is the object name, grid_21_5_2021_dv is the grid name. This folder contains some example output of running ```co_bandhead_grid.py```:
    - <or>**wvl_re.npy**</or> -- The wavelength array to each model in micron. This array is stored once.
    - all other <or>**\*.npy**</or> files -- Each file contains a 2D ```numpy``` array with the normalized model flux *per inclination*. The filename is derived from the set of all (six) other free parameter values. The ```filename_co_grid_point``` function in ```config.py``` defines the filename and can be used to find or iterate over models for further analysis (e.g., fitting). The model flux per inclination can be converted to, for example, a ```python``` dictionary with the following function: 
      ```python
      import numpy as np
       
      def read_output_per_inc_dict(filename, full_path, inc_array=None):
      """
      Read the numpy file which stores the normalized bandheads per inclination.
      
      :param filename: (str) the name of the model file.
      :param full_path: (str) path to the folder where <filename> is stored.
      :param inc_array: (array like) list or array of the inclinations used to calculate the grid.
      :return: a dictionary with keys inclination and values the model spectra.
      """     
      if inc_array is None:
         inc_array = np.array([10,20,30,40,50,60,70,80]) 
      results = np.load(full_path + filename + '.npy')
      dict_out = {}
      for j, i in enumerate(inc_array):
          dict_out[i] = results[j, :]            
      return dict_out
      ```
    
    -  <mag>**out.txt**</mag> -- This file contains a list of all the models that were calculated and specifies if they are saved or discarded: only models that result in "realistic" normalized fluxes are saved. This feature was implemented to save storage for large grids. The ```maxmin``` parameter of the function ```run_grid_log_r``` in ```flat_disk_log_grid.py``` is used to define which models are saved/discarded.


## Figures
The plot-ready files for the most important figures in the paper can be found in the folders <green>**fig_1**</green>,  <green>**fig_2**</green>, and <green>**fig_4**</green> described below. 

The best fit SED model parameters, the photometry and the Kurucz models for Figure 3 can be found in the <green>**software/aux_files**</green> folder described above. The function ```load_inc_dust_dict``` in ```config.py``` can be used to load the <or>**\*.pkl**</or> files, as for example in line 321 of ```flat_disk_log_grid.py```, where from the pickled ```python``` dictionaries the best fit SEDs are reproduced using ```sed_calculations.py```. 

Remaining figures can in principle be reproduced using the provided code (Figures 5, B1 and C1) and/or information present in the paper (Figure 6).
- <green>**fig_1**</green>: All plotted models saved as <or>**\*.npy**</or> files and named with the relevant parameter and its value. The wavelength array (in &mu;m) is the same for all and called <or>**model_wavelength_array.npy**</or>.

- <green>**fig_2**</green>: Models saved in the same way as above. The file <or>**data.npy**</or> contains a 2D array with the B163 spectrum, which can be loaded as follows:
  ```python
  import numpy as np 
  wvl_B163, flux_B163 = np.load("fig_2/data.npy", allow_pickle=True)
  ```

- <green>**fig_4**</green>: The <or>**\<obj>.npy**</or> files each contain five arrays, which can be loaded as follows:
  ```python
  import numpy as np 
  wvl_obj, flux_obj, wvl, conv_flux, fit_mask = np.load("fig_4/"+obj+".npy", allow_pickle=True)
  ```
  where ```wvl_obj``` and ```flux_obj``` are the data wavelength (in &mu;m) and normalized flux (these are the same spectra provided in the data folder); ``` wvl``` and ```conv_flux``` are the model wavelength (in &mu;m) and normalized flux for the best fit ; and ```fit_mask``` is a mask to the data arrays that marks the data included in the fitting. 
    
The provided files generally include the entire NIR spectrum or modeled wavelength range, not only the plot ranges on the figures in the paper. As such, the model files can also be used to check output obtained from running the code.

## Extra Information

The code as included in this package under <green>**software**</green>  does not include the feature of calculating the bandheads for 13CO. The version which does include this feature  is available on GitHub at https://github.com/JPoorta/co_bandhead_disk_model.git, where the code will be maintained and developed for further projects. The version uploaded in this package is simply the exact version that was used to generate the 12CO model grid analyzed in the paper. The 13CO models where calculated later for selected gridpoints and added to the 12CO models. 

