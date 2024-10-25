# OBRWR

This is the github repository for the Optimally Biased Random Walk with Restart Workspace.
Find here the corresponding data folder (NODES_RAWDATA & NODES_Project_Zenodo).
The containers mount volumes based on the similar arborescence of NODES_Project_Zenodo and this repository.

All containers provide either JupyterLab or Rstudio: when you are running a container you will be able to access it through\\
your favorite web-browser at the adress :

localhost:XXXX where XXXX is the port defined in the docker.sh 



## Docker
The only dependency is Docker.
See https://docs.docker.com/engine/install/

## Importing container images
In order to run the containers you will have to import the images from NODES_Project_Zenodo/container_images

```console
foo@bar:$ docker load < <file>.tar
```

## Root initrc
We define environment variables in the initrc at the root of the repository:
```
export ROOT_PROJECT_PATH="/mnt/project/NODES/NODES_Project_Zenodo" <-- path to root of data folder
export PATH_WORKSPACE="/mnt/project/NODES/OBRWR" <-------------------- path to here
export PATH_OBRWR="${PATH_WORKSPACE}/obrwr_python" <------------------ you should not have to touch this
export PROJECT_PATH=$ROOT_PROJECT_PATH <------------------------------ nor this
```
Please make sure the paths are set as necessary.
We assume a Unix system is being used.
All other initrc should not be modified.

## Running a particular container
From a folder with a docker.sh/docker_method.sh and initrc.\\
Take a look at docker.sh
```console
foo@bar:$ cat docker.sh
sudo docker run -dp 8892:8888 \
        -w /work -v $PROJECT_PATH"/00_InputData:/work/00_InputData" \
        -v $PROJECT_PATH"/05_Output:/work/05_Output"\
        -v $PROJECT_PATH"/01_Reference:/work/01_Reference"\
        -v $PWD"/03_Script:/work/03_Script"\
        -v $PATH_OBRWR"/03_Script:/opt/conda/lib/python3.10/site-packages/obrwr"\
        --name jup-highs-cibnos\
        jupyter-highs

sleep 1s

sudo docker exec jup-highs-cibnos jupyter server list
```
The docker.sh bash script defines:
  - The port mapping -dp XXXX:YYYY (remember the XXXX)
  - The folders which should be mounted as volumes -v
    
        - the $PROJECT_PATH and $PATH_OBRWR environment variables will be set by sourcing the local initrc
    
        - It automatically mounts obrwr as a python package
    
  - The name of the container --name docker_name
    
It then waits a bit (sleep 1s).

And then asks the container to display the url where the server is accessible.

We can go ahead and run both initrc and docker.sh :
```console
foo@bar:$ source initrc
foo@bar:$ sh docker.sh
http://aaaaaaaaa:YYYY/?token=##LONGTOKEN##
```
The url displayed here is from the point of view of the container. We really just need the token.
In your browser access :

localhost:XXXX/?token=##LONGTOKEN##

Where XXXX is defined in docker.sh.

And you will have a working environment to run the different notebooks/Rmarkdown files in 03_scripts folder.

## If url/token does not show 
Then this should do it, where the docker_name is defined in docker.sh:
```console
foo@bar:$ sudo docker exec docker_name jupyter server list
http://aaaaaaaaa:YYYY/?token=##LONGTOKEN##
```
