FROM jupyter/base-notebook:latest
USER root
RUN apt update
RUN apt install -y git
RUN git clone --depth 1 --branch v1.6.0 https://github.com/ERGO-Code/HiGHS
WORKDIR /home/jovyan/HiGHS
RUN apt install -y cmake
RUN apt install -y build-essential
RUN mkdir build && cd build && cmake ..
WORKDIR /home/jovyan/HiGHS/build
RUN make
RUN make install
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/HiGHS/build/lib/
WORKDIR /home/jovyan/HiGHS
RUN pip3 install ./
WORKDIR /home/jovyan
RUN ldconfig
RUN ls
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir /opt/conda/lib/python3.10/site-packages/obrwr
WORKDIR /home/jovyan
USER 1000
