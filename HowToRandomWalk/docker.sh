sudo docker run -dp 8894:8888 \
	-w /work -v $PWD"/00_InputData:/work/00_InputData" \
     	-v $PWD"/05_Output:/work/05_Output"\
	-v $PWD"/01_Reference:/work/01_Reference"\
	-v $PWD"/03_Script:/work/03_Script"\
	-v $PATH_OBRWR"/03_Script:/opt/conda/lib/python3.10/site-packages/obrwr"\
	--name jup-highs-howto\
	jupyter-highs

sleep 1s

sudo docker exec jup-highs-howto jupyter server list
