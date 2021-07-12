# Dynamic model training

Для решения задачи оптимального управления роботом, требуется точная динамическая модель, но получить ее аналитическими способами не всегда возможно. В данном репозитории представлен инструмент для получения динамической модели робота, нейросетевыми методами. Нейросеть предсказывает следующее состояние робота через заданную временную дельту, на основании текущего состояния и управляющего воздействия.

## Overview

В репозитории созданы следующие инструменты:
1. [Классы и утилиты для обучения модели](/docs/trainer.md)
2. [Пример обучения модели дифференциального робота](/docs/train_rosbot_model_example.md) 
3. [Инструкция по самостоятельному обучению модели](/docs/train_model.md)
  * Описание класса датасета и как собирать данные для обучения
  * Описание класса модели робота
  * Запуск обучения и анализ результатов
  * Скрипт для экспорта модели из Pytorch в ONNX


### 2.1 Минимальные системные требования
* 4 ядра процессора
* 8 ГБ ОЗУ
* Видеокарта nvidia (например Nvidia GTX 1060 ti)

## Start Guide

1. Собрать докер контейнер и убедиться, что Pytorch нормально запускается на GPU (простой тест)
2. Запустить обучение модели на тестовых данных, собранных для робота Rosbot в симуляторе Gazebo 9
3. Переопределить классы датасета и модели робота и создать скрипт обучения
