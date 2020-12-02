## Project Summary

The purpose of ENSO_forecast is to propose new methods based on machine learning predicting El Niño-Southern Oscillation (ENSO) with time and computation efficiency and Pearson correlation can achieve greater than 0.6. Traditional theory-based models are too computationally expensive for predicting ENSO. Our forecasting system not only can maintain skillful prediction with a Pearson correlation above 0.6 for long-range forecast but also can let clients run our forecasting system quickly and on a local computer when out in the field, and can allow for a quick visualization of the comparison with instrumental data and theory-based models.

For more detail of the project, please see and download the PDF version: [Project Report] (缺 PDF url)

## Planned the project 
**Gantt Chart**
![image](https://ds-560.github.io/ENSO_forecast/gantt_chart_week1.jpg)

For more details see [ Download Gantt Chart of ENSO_forecast](https://ds-560.github.io/ENSO_forecast/PROJECT_GANTT_CHART.xlsx).

**Milestone**
- Electing a project manager for our team 
- Deliver the sildes for the 1st meeting
- Meet with our instructor (1st meeting) 
- Create and deliver the sildes for the 2nd meeting
- Meet with our instructor (2nd meeting) 
- Meet with our instructor (3rd meeting)
- Meet with our instructor (4th Meeting)
- Deliver the rehearsal video
- Final meeting
- Assemble and publish the code
- Report submission

**Deliverables**
- Arrange tasks for each person for the 1st meeting
- Create and beautify the sildes
- Resolve Datetime format error
- Test the code on different environment
- Put experimental results on figshare

## Identified Wastes
- Skills: The user needs to learn how to run our codes in the terminal and understand Github’s tutorial.
- Waiting: The user needs to wait for a couple of minutes to get the result.
- Over-processing: The user needs to input parameters from the our README.md in the terminal each time when they use our system.
- Inventory: Multiple folders in the local, so the user might need to find the results in those folders.

## Usage of ENSO_forecast
See our GitHub [README.md](https://github.com/DS-560/ENSO_forecast/blob/main/README.md)

## Used Packages
- Pytorch
- Scikit-learn
- Xarray

## Used Dataset for ENSO_forecast
Download the data through the link mentioned below directly on a local computer that used for the ENSO_forecast project

- [COBE-SST2](http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc)
- [Nino3.4 Index](http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/nino34.long.anom.data.txt)
- [MPI](http://portal.nersc.gov/project/dasrepo/AMS_ML_Tutorial/MPI_tas_anomalies_regridded.nc)
- [CNRM-CM5](http://portal.nersc.gov/project/dasrepo/AMS_ML_Tutorial/CNRM_tas_anomalies_regridded.nc)
- [Reforecast Data (GRIB format)](https://drive.google.com/file/d/1I5-zCzZgjZjfCAEPfSZaChRw0BHEdooP/view?usp=sharing)
- [Reforecast Data (NetCDF format)](https://drive.google.com/file/d/1fW3Dbm3DAPjIb64AlN4kjctwj3rRsDmW/view?usp=sharing)


## Contact Infromation of Our Team
- Zerui Xie: _zeruixie@usc.edu_
- Jieqiong Pang: _jieqiong@usc.edu_
- Kuan-Hui Lin: _kuanhuil@usc.edu_  
- Yunyi Liao: _yunyilia@usc.edu_
- Jinhong Lei: _leijinho@usc.edu_
- Feilong Wu: _feilongw@usc.edu_
