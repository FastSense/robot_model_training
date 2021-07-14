Для обучения моделей рекомендуется работать в докере. В предоставленном образе уже есть все необходимые пакеты.

## Установка docker в систему

Чтобы работать с GPU, нужен nvidia-docker.

1. Установить nvidia-docker по [инструкции](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Добавить себя в группу `docker` по [инструкции](https://docs.docker.com/engine/install/linux-postinstall/). Запуск любых команд для докера не должен требовать `sudo`
3. Проверить, что докер работает без `sudo`:
```
docker run hello-world
```

## Использование докера

1. Сборка образа:

```
cd docker
./build.sh
```

2. Запуск контейнера:

```
./run.sh
```

3. Теперь, чтобы подключиться к контейнеру, выполнить:

```
docker attach pytorch-container
```

Этот репозиторий примонтируется в папку `~/ws` внутри контейнера.

Проверьте, что GPU доступна:
1. Команда `nvidia-smi` должна показать видеокарту (или несколько, если есть).
2. В консоли `python3` выполните две строки:
```
import torch
print(torch.cuda.is_available())
```
Должно напечатать `True`.