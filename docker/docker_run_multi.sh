data_path=$1

docker run -it --gpus all \
    --name TES \
    --shm-size=128G \
    -v $data_path:/home/data\
    -v $PWD:/home/ws/TESNet\
    tes:v0.1 bash