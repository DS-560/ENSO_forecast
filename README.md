python train.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --name test_cnn --epoch 2 --startdate 1980-01-01 --enddate 2101-12-31   --dataset CNRM

python test.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --name test_cnn.pt  --startdate "" --enddate ""  --test_start 2002-01-01 --test_end 2015-12-31 --dataset observations --compare_ground_truth

python test.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --dataroot1  "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --dataset CNRM --name linear_classification --startdate 1950-01-01 --enddate 2050-12-31 --instrument_data "./datasets/nino34.long.anom.data.txt" --test_start 2002-01-01 --test_end 2015-12-31  --model linear_regression --classification --compare_ground_truth

python test.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --name NinoPrediction_leadtime2_timespan1850-01-01-2299-01-01.pt  --startdate "" --enddate ""  --test_start 2000-01-01 --test_end 2010-12-31 --dataset observations --classification True --leadtime 2 --num_input_time_steps 2 --compare_ground_truth

python reforecast.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --name NinoPrediction_leadtime2_timespan1850-01-01-2299-01-01.pt  --startdate "" --enddate "" --test_start 1992-01-01 --test_end 2015-12-31   --dataset observations  --leadtime 2 --reforecast_data reforecast.txt --compare_ground_truth --num_input_time_steps 2

python reforecast.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --dataroot1  "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --dataset CNRM --name LR   --startdate 1950-01-01 --enddate 2050-12-31 --test_start 1992-01-01 --test_end 2015-12-31  --leadtime 2 --reforecast_data reforecast.txt --model linear_regression --compare_ground_truth --classification

python test.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "" --name NinoPrediction_leadtime2_timespan1850-01-01-2299-01-01.pt  --startdate "" --enddate ""  --test_start 2000-01-01 --test_end 2010-12-31 --dataset observations  --leadtime 2 --num_input_time_steps 2 

python test.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --dataroot1  "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --dataset CNRM --name linear_classification --startdate 1950-01-01 --enddate 2050-12-31 --instrument_data "./datasets/nino34.long.anom.data.txt" --test_start 2002-01-01 --test_end 2015-12-31  --model linear_regression --classification --compare_ground_truth


# ENSO_forecast

## Usage
### Use Microsoft Windows 10 as the operating system.
- Clone this repo using Git Bash:
```bash
git clone https://github.com/FeilongWu/ENSO_forecast.git
```

If you have pretrained models and datasets, put them under directores "/checkpoints" and "/datasets", respectively.
### All the commands below are executed using Command Prompt unless specified otherwise.
- Set the repository you just cloned as your working directory. The path to the directory may vary for different users. An example command is shown below:
```bash
cd "C://Users//your//name//ENSO_forecast"
```

- If you have <strong>venv</strong> installed, please skip. Use the following command to install  <strong>venv</strong>. This requires Python3.8:
```bash
pip install --user virtualenv
```
- Create a virtual environment (named ENSO) and activate it:
```bash
python -m venv ENSO
.\ENSO\Scripts\activate
```
- Install dependency(ies) for the environment.:
```bash
pip install -r requirements.txt
```
- You can extract a dependency list:
 ```bash
pip freeze > requirements.txt
```
- To see the descriptions of parameters for training/testing/reforecasting, you can use help function to print the descriptions on screen. An example of printing descriptions for training is given below.
 ```bash
python train.py -h
```
### Training
- You must specify "dataroot", "name", "startdate", and "enddate", which refer to the path to the training dataset, the name of your CNN model, the training starting date, and the training end date, respectively. An example command to train a CNN model with CNRM dataset is given below. The linear regression model will be trained in runtime when trying to do predictions.
 ```bash
python train.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --name test_cnn --epoch 2 --startdate '1980-01-01' --enddate '2101-12-31'   --dataset CNRM
```

### Testing
- Specify the four required parameters "dataroot" and "name" for training. At this time, the "name" parameter refers to the path to a CNN model by which you want to preduce predictions. It is the name of the experiment for using linear regression. The "test_start " and "test_end " are the start and end dates for prediction. Specify "startdate" and "enddate" as empty string for using CNN, but they should be the start and end training dates for using linear regression. The "dataroot" refers to the file of predictors for CNN and is the training file for linear regression. The "dataroot1" is the path to the file of predictors for linear regression and "instrument_data " is the path to the ground truth Nino3.4 index as a text file for using all models. A text file containing the predictions and a plot will be saved under ./results/. The following command initiates a testing by using a pretrained model.
 ```bash
python test.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --name test_cnn.pt  --startdate "" --enddate ""  --test_start 2002-01-01 --test_end 2015-12-31 --dataset observations --compare_ground_truth
```
- By default, the testing would not produce a result in comparison with ground truth. If "--compare_ground_truth" is not used, the resulting plot will only contains the model predictions and there is no need to specify ground truth Nino3.4 index as in "instrument_data ". The following code produces predictions using linear classification.
```bash
python test.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --dataroot1  "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --dataset CNRM --name linear_classification --startdate 1950-01-01 --enddate 2050-12-31 --instrument_data "" --test_start 2002-01-01 --test_end 2015-12-31  --model linear_regression --classification 
```
### Reforecast
- Model predictions can be compared with theory-based model predictions if reforecast data is available. The following command is to use predictions from a CNN pretrained model against reforecast data. Specify your reforecast data path and the year to which you want to extract in the "reforecast.txt" separated by a comma.
```bash
python reforecast.py --dataroot "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --name NinoPrediction_leadtime2_timespan1850-01-01-2299-01-01.pt  --startdate "" --enddate "" --test_start 1992-01-01 --test_end 2015-12-31   --dataset observations  --leadtime 2 --reforecast_data reforecast.txt --compare_ground_truth --num_input_time_steps 2
```
- You can also use linear classification model to test its ability against reforecast data.
```bash
python reforecast.py --dataroot "./datasets/CNRM_tas_anomalies_regridded.nc" --dataroot1  "./datasets/sst.mon.mean.trefadj.anom.1880to2018.nc" --instrument_data "./datasets/nino34.long.anom.data.txt" --dataset CNRM --name LR   --startdate 1950-01-01 --enddate 2050-12-31 --test_start 1992-01-01 --test_end 2015-12-31  --leadtime 2 --reforecast_data reforecast.txt --model linear_regression --compare_ground_truth --classification 
```
