sudo docker run -dp 8788:8787 \
	-w /work -v $PROJECT_PATH"/00_InputData:/home/rstudio/work/00_InputData" \
     	-e PASSWORD="pwdroot"\
	-v $PROJECT_PATH"/05_Output:/home/rstudio/work/05_Output"\
	-v $PROJECT_PATH"/01_Reference:/home/rstudio/work/01_Reference"\
	-v $PWD"/03_Script:/home/rstudio/work/03_Script"\
	--name HPN\
	hpn

sleep 1s

