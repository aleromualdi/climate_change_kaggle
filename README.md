# Climate Change Analysis (Kaggle)

This repository contains the code for the analyis of the dataset contained in the Kaggle competition "Climate Change: Earth Surface Temperature Data". <br>

Plese download the the datasets `GlobalLandTemperaturesByCity.csv​`,`lobalLandTemperaturesByCountry.csv​` from https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data, and place them into the `data/` folder.


# Repository structure

    ├── data                    # please add input data
    ├── output                  # contains generate outputs
    ├── analysis                # contains the scripts for generating the solution
    └── models.py               
    └── README.md


# Requirements

Please install the requirements with 

```
pip install -r requirements.txt
```


# Description

The `analysis/` fodler contains scripts to generate the solution, including the ipython notebook `1_eda.ipynb` for exploratory data analysis, and the script `2_predictive_modeling.py` for training models for average temperature change prediction on cities where high temperature variability is observed in past years. <br>

Please first run the notebook `1_eda.ipynb` in order to procedd the year-aggregated average temperature epr city. <br>

To forecasts the average temperature across years, I trained a support vector regression model (SVR), and a multilayer dense nerual network model. <br>
The input data is preprocessed using a look back function of slide 1. The function defines the number of recent data points to be used when predicting each future value in the time series. <br>
The choice of slide 1 is motivated by the sudden increase of average temperature after 1980. Larger slide would make the model less sensitive to local change in average temperature.<br>
The input data is then split begtween train and test sets, using the date 1.12.1950 as splitting time point.


The results show only little agreement with the ground trugh, and only for the city of `Heihe` SVR model could predict the temperature trend. In other cases, it seems that the sudden increase in averate temperature from 1980 maked the prediction hard.<br>

The model could be improved by including observation from other variables (multivariate model), for example pollution indices, that are known to correlate with the avergae temperature increase.<br>

More robust models could be buind upon Long short-term memory (LSTM) nodes arhcitecture. Such models can predict time series given time lags of unknown duration, and are relative insensitive to gap length.<br>

