## 1. Install docker
Follow the instructions here [Docker-install](https://docs.docker.com/desktop/install/linux/ubuntu/)
## 2. Install NVIDIA Container Toolkit
Follow the instructions here [NVIDIA-install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 3. Go to the project path
```bash
cd {PATH_TO_THIS_PROJECT}/TemporalEventStereo
```
## 4. Build Docker Image
Build docker image using the following script.
This will build the docker image "tes:v0.1"
```bash
source docker/docker_build.sh 
```
## 5. Start Docker Container

```bash
source docker/docker_run_multi.sh {ABSOLUTE_DATA_DIR_PATH}
```