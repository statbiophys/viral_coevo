# README

This repository contains the scripts to generate the figures of the paper: **Chardès et al., Evolutionary stability of antigenically escaping viruses**.
Arxiv link: https://arxiv.org/abs/2304.11041

The folder `src` contains the core scripts to simulate antigenic viral waves escaping from the immune system as described in the paper.
Those scripts contain python classes that call functions in `numba_functions.py` in which the numba library is used to increase the computational speed.

All the other folders are associated with figures of the paper as specified by the label.
With the only exception of `fig1_sketch`, where data generation and plotting are contained in the same python notebook and are quick to generate, all the other folders are structured as follows:
- One or more python notebooks starting with the name `data_gen_[...]` use the scripts in the `src` folder to generate all the necessary data that is stored in the sub-folder `data/`. Those data are usually `.tsv` tables or `.pickle` binary files.
- One or more notebooks for each panel, which read the information in the `data/` folder and generate the plots exporting them as `.svg` files in the `plots/` sub-directory.
- The `.svg` figures contained at the same directory level of the python scripts, typically called `figure[...].svg`, are inkscape assemblies of the plots exported at the previous step.



