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
