# Gaussian Mixture Modeling for Flow Cytometry Data fcGMM

## Install requirements
To install requirements, first make virtual enviroment and activate it
```
python  -m venv venv
.\venv\Scripts\activate
```
Then, install requirement with pip
```
python -m pip install -r requirements.txt
```

## Notebook or script mode

This git repository allows either to run a Jupyter notebook 
or a script. The motebook allows the analysis of a single fcs file the script allows the batch execution on more fcs files.

## Notebook mode

To run the script run Jupyter and open the `ExampleNotebook.ipynb`

## Script mode

### Set up files

![plot](./mixGaussResult.png)

The program requires the existence of a file descirbing were the cytofluorometer outpufiles are placed. 

```
data/
AutoFl:dataAF.fcs
0:data00.fcs
18:data18.fcs
24:data24.fcs
42:data42.fcs
48:data48.fcs
```
The name of the file may be something like `fileCTV-PKH.dat`, where `CTV` and `PKH` are the names of the fluorescence channel.

At the first execution we will run the program with the preprocessing flag:
```
python runFcGMM.py --preprocessing --dim 2 -i PKH-CTV
```
The `--dim` flag sets the number of dimensions that are taken in consideration in the Gaussian MIxture Model which can be from 1 to 3.
The `--preprocessing` flag allows to set the threshold for forward scattering (FSC-A) and
side scattering (SSCA). It also allows to threshold outliers using the estimate of the density function of the scatter plot computed using a Gaussian kernel.

![plot](./preproc.png)

Then to set the inital values of the Gaussian Mixture Model we run:
```
python runFcGMM.py --setInit --dim 2 -i PKH-CTV

```
![plot](./initVals.png)

As it is know the Gaussian Mixture Model relies heavily on the initial conditions that are set at the beginning of the Expectation Maximization algorithm. The `--setInit` flag opens an interactive window that allows us to set the initial values of centrists and the variances of the single Gaussian distributions forming the Gaussian Mixture Model.

Finally, to run the EM Gaussian Mixture Model we run:
```
python runFcGMM.py --dim 2 -i PKH-CTV
```
## Citation

This code was used in the following paper, if you find this project useful, please cite:
> Giovanna Peruzzi, Mattia Miotto, Roberta Maggio, Giancarlo Ruocco, and Giorgio Gosti.
> *Asymmetric binomial statistics explains organelle partitioning variance in cancer cell proliferation*.
> Commun Phys 4, 188 (2021).
> https://doi.org/10.1038/s42005-021-00690-5


