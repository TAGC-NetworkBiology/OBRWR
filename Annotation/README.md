You can find the corresponding datafolder here : https://doi.org/10.5281/zenodo.14265393

TO PERFORM ANNOTATION

    The jupyter-scientific-blast docker image should be loaded (see README at root of repository).
    Make sure the Annotation data folder is accessible in the corresponding Project folder
    And that the initrc at the root of this workspace is instantiated (see README at root of this repository)
    here define the variables : source initrc
    and after conveniently changing the docker.sh file (see README at root of this repository) : sh docker_obrwr.sh
    Then from inside the jupyter lab run the scripts in order. Blast has been run on a cluster.
