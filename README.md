# Plastic landmark anchoring in zebrafish compass neurons
This repository hosts scripts (jupyter notebooks) to analyze data for our manuscript **Plastic landmark anchoring in zebrafish compass neurons** soon to be published (the preprint version of the study can be found [here](https://doi.org/10.1101/2024.12.13.628331)).

## Data
Preprocessed data to be analyzed by the scripts here are deposited on [Zenodo](https://doi.org/10.5281/zenodo.17233579).

## Requirements
Please install the packages required to run the scripts from the provided environment file by running `conda env create -f environment.yml`.
Activate the installed conda environment and register it as a kernel for jupyter by running `python -m ipykernel install --user --name=landmark`.

## Outputs
The notebook save the figures as SVG (in the `svg` directory) as they appear in the paper figures.
