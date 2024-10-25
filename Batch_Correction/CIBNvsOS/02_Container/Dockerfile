# syntax=docker/dockerfile:1
FROM jupyter/base-notebook:latest
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends dirmngr gnupg2 apt-transport-https ca-certificates software-properties-common
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get install -y r-base build-essential
COPY R_requirements.R R_requirements.R
RUN apt-get install -y --no-install-recommends cmake
RUN R < R_requirements.R --no-save
COPY R_jupyter.R R_jupyter.R
RUN apt-get install -y --no-install-recommends openssl libcurl4-openssl-dev libssh-dev libxml2 libfontconfig1-dev curl
#RUN R < R_jupyter.R --no-save
RUN R -e 'install.packages(c("credentials"))'
RUN apt-get install -y --no-install-recommends libtiff-dev libfreetype-dev libharfbuzz-dev libfribidi-dev
RUN R -e 'install.packages(c("textshaping"))'
#RUN R < R_jupyter.R --no-save
#RUN R -e 'install.packages(c("rmarkdown"))'
#RUN R -e 'install.packages(c("devtools"))'
RUN R < R_jupyter.R --no-save
RUN apt-get install -y --no-install-recommends git
RUN R -e 'devtools::install_github("IRkernel/IRkernel")'
RUN R -e 'IRkernel::installspec()'
RUN chmod -R 777 /home
USER 1000
RUN pip3 install ipympl
