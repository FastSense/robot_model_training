# PyTorch
Докер для работы с pytorch.  

Пример сборки:

```
docker build -t torch1.6 -f ubuntu18.04_torch1.6 .
```

Пример запуска:

```
docker run --gpus 1 -it --rm torch1.6
```



## Docker 

Рекомендуется работать через Docker. 

Для работы в Gazebo сборка образа и запуск контейнера осуществляется из директории docker: создается `pytorch-image` образ и запускается `pytorch-container` контейнер. 


### Install Guide
- Install docker  https://docs.docker.com/engine/install/ubuntu/
- Docker post install steps https://docs.docker.com/engine/install/linux-postinstall/
```
./docker/dependencies.sh		# установка зависимостей на локальную машину
./docker/build.sh			      # Build image 
./docker/run.sh				      # Create & Run container
```

### Usage Guide
```
docker start pytorch-container 		 
docker attach pytorch-container   		 
```

## Компиляция и запуск симуляции

```
docker attach $container_name

```