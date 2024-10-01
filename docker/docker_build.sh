export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

docker build --build-arg UID=$HOST_UID --build-arg GID=$HOST_GID \
    -t tes:v0.1 \
    -f docker/Dockerfile .
