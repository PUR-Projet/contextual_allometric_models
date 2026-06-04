# Contextualizing Pan-Tropical Allometric Models for Biomass Estimation

This repository contains the original code of the experiments of the paper
[Contextualizing Pan-Tropical Allometric Models for Biomass Estimation](https://doi.org/10.64898/2025.12.16.694295) 

## Setup

Run the following commands to setup a python environment.

```
pyenv local 3.11.14
pip install poetry --user
poetry config virtualenvs.in-project true
poetry lock
poetry install --no-root --all-extras
```

## Contribution #1 : More precise Contextual Pan-Tropical Allometric Models

Command to launch experiments:

`poetry run python minimal_cofarm.py`

Each baseline and proposed model will be run on the same 30 random train/test splits. This will take some time, depending on your computer.

The code outputs a file `results_LogReg-LogReg-NN-COFARM-COFARM-NN-HGBRT-ContextualChave.csv` that contains all metrics for each model and confidence intervals. This corresponds to Table 3 of the paper.

## Contribution #2 : Predicting Additional Estimation Error when applying Allometric Equation to New Sites

Command to launch experiments with synthetic shifts:

`poetry run python minimal_bound.py `

You should see an output similar to:
```
>>> gathering training data
>>> training predictor
train data: (50000, 5) (50000,)
scaler [0.3982765  0.         0.32533303 0.04021943 1.        ] [0.16670886 1.         0.54697324 0.32371542 0.        ]
linreg [-18.7463939   13.59471261  45.20913574 154.72572956   0.        ] 68.45591549862023
train sign Accuracy: 0.86528
Pipeline(steps=[('scaler', StandardScaler()), ('linreg', RidgeCV())])
train R2: 0.8403201579865209
>>> evaluating delta mae predictor on altitude-low
>>> evaluating delta mae predictor on altitude-high
>>> evaluating delta mae predictor on drymonths-few
>>> evaluating delta mae predictor on drymonths-many
>>> evaluating delta mae predictor on rainfall-low
>>> evaluating delta mae predictor on rainfall-high
>>> evaluating delta mae predictor on foresttype-dry
>>> evaluating delta mae predictor on foresttype-wet
>>> evaluating delta mae predictor on foresttype-moist
>>> evaluating delta mae predictor on foresttype-mangrove
>>> evaluating delta mae predictor on continent-asia
>>> evaluating delta mae predictor on oldgrowth-0
>>> evaluating delta mae predictor on random-1
>>> evaluating delta mae predictor on random-2
test R2: 0.8369188393170234
```

The code outputs Figure 5 of the paper as [bound_results.png](bound_results.png) and Table 2 as [bound_results.tex](bound_results.tex). 

## How to use the models

You can ask for AGB predictions from pre-trained models with the following command:
`poetry run prediction.py -i example_input.csv -o example_output.csv`

This will read `example_input.csv` with the same format as the [CSV file](chave.csv) from Chave et al and 
write `example_output.csv`with the same columns and an `AGB` column that contains predictions from the model.

Mandatory columns are the following:
 - DBH (in cm)
 - H (in m)
 - WD (in g.cm^-3)
 - Continent (Asia/Americas/Africa)
 - Rainfall (annual, in mm)
 - OldGrowth (1/0)
 - ForestType (Dry/Moist/MoistMangrove/Wet)
 - DryMonths (0-12)
 - Altitude (in m)

You can select a model with the `-m` option. Provided pre-trained models include: 
- `LogReg`
- `LogReg-NN`
- `COFARM`
- `COFARM-NN`
- `ContextualChave`

By default the `COFARM-NN` model is used.

## Carbon Emissions of this Research

We estimate carbon emissions of this research using the [CodeCarbon](https://codecarbon.io/) project. Experiments were launched on an Apple M3 laptop in France.

The final run of the two contributions entail
  - 1.1896 g CO2eq (contextual pan-tropical model)
  - 0.0062 g CO2eq (transfer error model)
as measured by CodeCarbon v.3.2.6.


