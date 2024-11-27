You can find the corresponding datafolder here :
https://zenodo.org/records/14228715

**TO RUN OBRWR**

1) The **jupyter-highs** docker image should be loaded (see README at root of repository).
2) Make sure the CIBNvsOS is accessible in the corresponding Project folder
3) And that the initrc at the root of this workspace is instantiated (see README at root of this repository)
4) **here** define the variables : source initrc 
5) and after conveniently changing the **docker_obrwr.sh** file (see README at root of this repository) : sh docker_obrwr.sh

**TO RUN PHONEMeS**

1) The **hpn** docker image should be loaded (see README at root of repository).
2) Make sure the CIBNvsOS is accessible in the corresponding Project folder
3) And that the initrc at the root of this workspace is instantiated (see README at root of this repository)
4) **here** define the environment variables : source initrc 
5) and after conveniently changing the **docker_PHONEMeS.sh** file (see README at root of this repository) : sh docker_PHONEMeS.sh

**For both**

6) You should now be able to access the jupyter/R studio at the defined localhost:XXXX in your browser.
