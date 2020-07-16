### Overview
This repository contains the prototpye implementation of the proposed alignment approximation 
approach presented in the paper 
***Alignment Approximation for Process Trees*** 
by Daniel Schuster, Sebastiaan J. van Zelst and Wil M. P. van der Aalst.

Corresponding author: Daniel Schuster 
([Mail](mailto:daniel.schuster@fit.fraunhofer.de?subject=github-incremental_a_star_approach))


This prototype implementation is using a fork of the [pm4py library](https://pm4py.fit.fraunhofer.de). 


### Repository Structure
* The main proposed algorithm is implemented in 
`approximate_alignment.py`.
* In `experimental_setup/experiments.py` you can analyze the results from the conducted experiments and also re-run 
them. Please make sure that the corresponding event log files, which are publicly available, are placed in the folders 
`experimental_setup/bpi_ch_18` and `experimental_setup/bpi_ch_19`.  
[BPI Ch. 18 event log](https://data.4tu.nl/repository/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)  
[BPI Ch. 19 event log](https://data.4tu.nl/repository/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1)


### Installation
* Python 3.7
* For platform-specific installation instructions regarding PM4Py, please visit 
[https://pm4py.fit.fraunhofer.de/install](https://pm4py.fit.fraunhofer.de/install)