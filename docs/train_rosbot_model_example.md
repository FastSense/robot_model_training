# Пример запуска обучения динамической модели rosbot

### 1. Назначение

 Пример запуска обучения динамической модели rosbot

### 2 Скачать [данные](https://drive.google.com/file/d/1zbWuxToTtiBUcXPGtaFSP3kP6n6Lcoep/view)
Данные собраны, с помощью логирующей ноды [logger](https://github.com/urock/rosbot/tree/develop/logger), для робота RosBot в робототехническом симуляторе Gazebo
Формат хранения данных, представлен на рисунке ниже

![datasets](/docs/images/datasets.jpg)

### 2.1 (Опционально) Зарегистрироваться в [wandb](https://wandb.ai/) 
### 2.2 (Опционально) [PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### 3 Изменить [конфиг](../examples/gz-rosbot/configs/gz-rosbot_1.yaml)

Необходимо определить путь до директорий с 
* **train_data_path** - путь до тренировочных датасетов
* **val_data_path** - путь до валидационных датасетов
* **test_data_path** - путь до тестовых датасетов

(Опционально) При наличии проекта в wandb, определить:
* **project_name** - имя проекта
* **entity** - имя владельца

(Опционально) изменить гиперпараметры
* **main_metric** - название ключевой метрики
* **layers_num** - количество слоев 
* **hidden_size** - количество нейронов
* **activation_function** - функция активации (elu / relu)
* **num_epochs** - количество эпох
* **rollout_size** - размер проезда для обучения
* **learning_rate** - скорость обучения
* **plot_trajectories** - флаг, построение графиков
* **save_plot** - флаг, сохранение графиков

### 4 Запуск обучения
```bash
cd /../gz-rosbot
python3 rosbot_train.py -cfg $CFG_PATH -name $WANDB_NAME
```
Аргументы:
* $CFG_PATH - путь до конфига
* $WANDB_NAME - название сессии в wandb

### Где посмотреть графики 
Для анализа результатов рекомендуется использовать [wandb](https://wandb.ai/) 

### Где будет лежать модель
Вкладка "files", при использовании [wandb](https://wandb.ai/) 


### Инференс

Для преобразования pytorch модели в .onnx модель, следует использовать скрипт [to_onnx.py](../examples/gz-rosbot/to_onnx.py)
При создании объекта model, следует передать такие же гиперпараметры, как у обученной нейросети.

