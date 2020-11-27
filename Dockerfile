FROM python:3
FROM nvidia/cuda:10.1-base
RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda install -y pytorch torchvision -c pytorch
RUN pip install joblib==0.17.0
RUN pip install jupyter==1.0.0
RUN pip install jupyter-console==6.1.0
RUN pip install jupyter-core==4.6.3
RUN pip install jupyterlab==2.1.5
RUN pip install matplotlib==3.2.2
RUN pip install pandas==1.1.4
RUN pip install Pillow==8.0.1
RUN pip install scikit-learn==0.23.2
RUN pip install scipy==1.5.4
RUN pip install xarray==0.16.1
RUN pip install netCDF4==1.5.4

RUN mkdir ENSO_forecast
Add /reforecast.py /ENSO_forecast/
Add /test.py /ENSO_forecast/
Add /train.py /ENSO_forecast/
Add /reforecast.txt /ENSO_forecast/
Add /models.py /ENSO_forecast/
Add /create_dataset.py /ENSO_forecast/
Add /ENSO_forecast.ipynb /ENSO_forecast/
Add /README.md /ENSO_forecast/
Add /checkpoints /ENSO_forecast/checkpoints 
Add /datasets /ENSO_forecast/datasets
Add /options /ENSO_forecast/options
Add /results /ENSO_forecast/results
WORKDIR /ENSO_forecast
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]