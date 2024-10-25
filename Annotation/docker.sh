sudo docker run -dp 8890:8888 \
	-w /work -v $PROJECT_PATH"/00_InputData:/work/00_InputData" \
     	-v $PROJECT_PATH"/05_Output:/work/05_Output"\
	-v $PROJECT_PATH"/01_Reference:/work/01_Reference"\
	-v $PWD"/03_Script:/work/03_Script"\
	--name jupsci-annotation\
	jupyter-scientific-blast

sleep 1s

sudo docker exec jupsci-annotation jupyter server list
