# Самостоятельное обучение модели
### 1 Сбор данных
Для каждого проезда (пролета) требуется логировать, время, управление и состояние робота. Это можно длеать любым удобным способом, например ROS нодой для логирования - [logger](https://github.com/urock/rosbot/tree/develop/logger). Каждый проезд (это один датасет) может быть представлен в виде 3 файлов:  
* *time.csv* - файл с временными отметками (timestamps)
* *state.csv* - файл с состоянием робота 
* *control.csv* - файл с текущим управлением
  
Для собранных данных нет строгого формата, главное чтобы было удобно загрузить их в класс датасета
### 2 Опредлеение класса датасета
Класс для представления датасета, и по необходимости, инструменты для взаимодействия с ним.
**Обязательно должны быть атрибуты:**
* *data_t* (torch.tensor of shape [num_samples, 1]): тензор с последовательность временных отметок
* *data_x* (torch.tensor of shape [num_samples, robot_state]): тензор с последовательностью состояния робота
* *data_u* (torch.tensor of shape [num_samples, control]): тензор с последовательностью управляющих воздействий

Пример [RosbotDataset](../examples/gz-rosbot/rosbot_dataset.py)

### 3 Определение класса робота
Класс для представления модели робота

Должен наследоваться от `torch.nn.Module`

**Обязательные атрибуты:**
* `model` - (`torch.nn.Module` или класс отнаследованный от `torch.nn.Module`): нейосеть, которую будут обучать

**Обязательные методы:**
* `get_optimizer` - возвращает `optimizer` (`torch.optim`)
* `get_loss_fn` - возвращат loss функцию. Loss функция может быть определена отдельным классом (см. ниже)
* `update_state` - возвращает следующее состояние робота с использованием нейросетевой модели (атрибут `model`).
  Аргументы:
  * `state (torch.tensor of shape [num_smaples, robot_state])` - текущее состояния,
  * `control (torch.tensor of shape [num_smaples, control])` - текущее управление,  
  * `dt (torch.tensor of shape [num_smaples, robot_state] or float)` - временной шаг
* forward - вычисление, выполняемое при каждом вызове модели
* calc_metrics - расчет вспомогателных метрик (возвращает словарь, где ключ - нвазание метрики и поле это значение метрики )
* plot_trajectories - Построение графиков.
Аргументы:
predicted_traj (torch.tensor of shape [num_smaples, robot_state]) - предсказанная траектория
ground_truth_traj (torch.tensor of shape [num_smaples, robot_state]) - истинная траектория
Возвращает: matplotlib.pyplot.fig

Пример [RosbotModel](../examples/gz-rosbot/rosbot_model.py)
Пример [RosbotLinearModel](../examples/gz-rosbot/rosbot_linear_model.py)

### 4 Определение loss функции
Класс для расчета ошибки (loss). Данный класс нужен, так как для разных роботов loss может отличаться (например при расчете loss не обязательно учитывать каждый элемент вектора состояний). 
Должен наследоваться от torch.nn.Module
**Обязательные методы:**
* forward - вычисление, выполняемое при каждом вызове (расчет loss)

Пример [RosbotModelLoss](../examples/gz-rosbot/rosbot_model.py#L9)

### 5 Процесс обучения
Обучение в нейросети происходит в **Trainer.fit** Аргументы:
* model - модель робота 
* train_data - лист датасетов для обучения
* val_data - лист датасетов для валидирования
* epochs_num - количество эпох обучения
* batch_size - размер батча
* rollout_size - длина проезда для обучения
* main_metric - название ключевой метрики, по которой выбирается лучшая модель
* device - дейвайс на котором производятся вычисления(cuda или cpu)
* use_wandb - флаг, если true, графики и метрики будут логироваться в wandb

Пример [rosbot_train](../examples/gz-rosbot/rosbot_train.py)

### 6 Анализ результатов
Для анализа результатов рекомендуется использовать [wandb](https://wandb.ai/) 

### 7 (Опционально) Использование конфигов
Для быстрой смены гиперпараметров возможно использование конфиг параметров.
Пример конфига [gz-rosbot_1.yaml](../examples/gz-rosbot/configs/gz-rosbot_1.yaml)
Пример использования [rosbot_train](../examples/gz-rosbot/rosbot_train.py#L41)


