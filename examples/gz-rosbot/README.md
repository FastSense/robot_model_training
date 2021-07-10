# Пример запуска обучения динамической модели rosbot

# 1. Назначение
 Пример обучения динамической модели rosbot

# 2. Как запустить?
### 2.1 Установка [инструкция](https://github.com/FastSense/ML-Training/blob/rosbot-gazebo-model/rosbot-gazebo-model/README.md#2-%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0)

### 2.2 Скачать [данные](https://drive.google.com/file/d/1zbWuxToTtiBUcXPGtaFSP3kP6n6Lcoep/view)

// TODO Тут будет нормальная ссылка,  на архив с 3 папками (train, validation, test), в которые я разложу траектории

### 2.3 Изменить конфиг
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

### 2.4 Запуск обучения
```bash
cd /../gz-rosbot
python3 rosbot_train.py -cfg $CFG_PATH -name $WANDB_NAME
```
Аргументы:
* $CFG_PATH - путь до конфига
* $WANDB_NAME - название сессии в wandb

### Где посмотреть графики 

### Где будет лежать модель

### Инференс


