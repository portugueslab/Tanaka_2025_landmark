# Jupyter notebooks
Notebooks are organized by the dataset they analyze, and named after the figure (either `main` for main figures and `ed` for extended data) where the experiment that generated the dataset is introduced.

## List of notebooks
- `main01.ipynb`: Analyzes the initial "sun-and-bars" experiment as introduced in **Fig. 1**. Also generates supplementary analysis panels in **Extended Data Fig. 1, 2, 3**.
- `main02.ipynb`: Analyzes the "jump and noise" experiment in **Fig. 2**. Also generates panels in **Extended Data Fig. 1, 5**.
- `main03.ipynb`: Analyzes the "symmetry" experiment introduced in **Fig. 3**. Also generates panels for **Fig. 4, Extended Data Fig. 1, 7**.
- `main05.ipynb`: Analyzes the "ablation" experiment in **Fig. 5**. Also generates some panels in **Extended Data Fig. 1, 9, 10**.
- `ed03.ipynb`: Analyzes the "starfield" experiment in **Extended Data Fig. 3**.
- `ed04a.ipynb`: Analyzes the "stonehenge" experiment in **Extended Data Fig. 4**. Also generates panels in **Extended Data Fig. 1**.
- `ed04b.ipynb`: Analyze the "elevation" experiment in **Extended Data Fig. 4**.
- `ed06.ipynb`: Performs the simulation presented in **Extended Data Fig. 6** and detailed in **Supplementary Note 1**.
- `ed08.ipynb`: Analyze the habenula visual receptive field mapping experiment in **Extended Data Fig. 8**.
- `ed10.ipynb`: Analyze the "AHV cell" experiment in **Extended Data Fig. 10**.

## Data structure
For most datasets, the data directory contains folders each corresponding to each recording, named like `20230101_f0_XXX`. 

The recording directory typically contains suite2p-preprocessed imaging data `data_from_suite2p_XXX.h5`, a `json` file specifying the anatomical mask, scanning metadata (`scandta.json`), time stamps from the microscope (`time.h5`) and `behavior` directory, under which you can find stimulus metadata (`XXX_metadata.json`), raw behavioral data (i.e., ~200 Hz time traces of tail angles `XXX_behavior_log.hdf5`), and closed loop stimulus log (i.e., ~60 Hz time traces of scene orientations `XXX_stimulus_log.hdf5`).

In some cases where multiple recordings were systematically made from a single fish (`main05` and `ed10`), the recording directories are nested under fish directories.
