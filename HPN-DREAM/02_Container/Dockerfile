FROM rocker/rstudio:latest
COPY R_requirements.R R_requirements.R
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends coinor-cbc coinor-libcbc-dev
RUN apt-get install -y --no-install-recommends libglpk-dev
RUN R < R_requirements.R --no-save
RUN R -e 'BiocManager::install(c("decoupleR"))'
RUN R -e 'BiocManager::install(c("paxtoolsr"))'
RUN chmod -R 777 /home
