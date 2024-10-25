sudo docker run -dp 8892:8888 \
	-w /work -v $PROJECT_PATH"/00_InputData:/work/00_InputData" \
     	-v $PROJECT_PATH"/05_Output:/work/05_Output"\
	-v $PROJECT_PATH"/01_Reference:/work/01_Reference"\
	-v $PWD"/03_Script:/work/03_Script"\
	-v $PATH_OBRWR"/03_Script:/opt/conda/lib/python3.10/site-packages/obrwr"\
	--name jup-highs-cibnVos\
	jupyter-highs

sleep 1s

sudo docker exec jup-highs-cibnVos jupyter server list
