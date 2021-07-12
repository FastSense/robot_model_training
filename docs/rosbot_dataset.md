# Класс RobotDataset

Класс представляющий собой набор данных (датасет) о перещених робота. ( Удобно считать, что каждый объект класса RobotDataset это траектория )

### Обязательные поля

* data_x - последовательность состояний робота (dtype torch.tensor размера [num_smaples, robot_state])
* data_u - последовательность управляющих воздействий (dtype torch.tensor размера [num_smaples, robot_control])
* data_t - последовательных временных отметок (dtype torch.tensor размера [num_smaples, 1])
* device - cuda или cpu

### Обязательные методы
 
**Отсутствуют**

Пример класса RobotDataset