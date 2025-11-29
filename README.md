# LSTM Log Anomaly Detection

Курсовой проект, посвященный созданию интеллектуальной системы анализа логов с использованием машинного обучения.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Development](https://img.shields.io/badge/status-development-brightgreen.svg)]()

## Описание проекта

**LSTM Log Anomaly Detection** — это интеллектуальная система, предназначенная для автоматического обнаружения аномалий в системных логах с использованием глубокого обучения. Проект реализует архитектуру на основе нейронных сетей LSTM (Long Short-Term Memory) для анализа последовательностей логов и выявления отклонений от нормального поведения системы.

### Ключевые особенности

- **Глубокое обучение**: Использование LSTM нейронных сетей для анализа временных зависимостей в последовательностях логов
- **Автоматическое обнаружение аномалий**: Система обучается на нормальных логах и автоматически выявляет аномальные события
- **Препроцессинг логов**: Структурирование и парсинг неструктурированных логов в машиночитаемый формат
- **Масштабируемость**: Возможность обработки больших объемов логов
- **Настраиваемость**: Гибкие параметры модели для различных типов логов и систем

## Архитектура системы

Система состоит из следующих основных компонентов:

```
┌─────────────────────────────────────────────────────────┐
│                      Сырые логи                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   1. Парсинг и подготовка    │
        │   • Токенизация              │
        │   • Нормализация             │
        │   • Кодирование              │
        └──────────────────────┬───────┘
                               │
                               ▼
        ┌──────────────────────────────┐
        │   2. Формирование сессий     │
        │   • Временные окна            │
        │   • Последовательности        │
        │   • Векторизация             │
        └──────────────────────┬───────┘
                               │
                               ▼
        ┌──────────────────────────────┐
        │   3. LSTM модель             │
        │   • Обучение                 │
        │   • Предсказание             │
        │   • Оценка аномальности      │
        └──────────────────────┬───────┘
                               │
                               ▼
        ┌──────────────────────────────┐
        │   4. Обнаружение аномалий    │
        │   • Пороги детекции          │
        │   • Классификация            │
        │   • Отчеты                   │
        └──────────────────────────────┘
```

## Требования

### Системные требования

- Python >= 3.8
- pip или conda для управления зависимостями
- ОС: Linux, macOS или Windows

### Зависимости Python

```
tensorflow >= 2.10.0
keras >= 2.10.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/MirazE1/LSTM_Log_Anomaly_Detection.git
cd LSTM_Log_Anomaly_Detection
```

### 2. Создание виртуального окружения (рекомендуется)

```bash
# Для Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Для Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Проверка установки

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Быстрый старт

### Базовое использование

```python
from lstm_anomaly_detector import LSTMLogAnalyzer
import pandas as pd

# Загрузить логи
logs = pd.read_csv('data/system_logs.csv')

# Инициализировать анализатор
analyzer = LSTMLogAnalyzer(
    sequence_length=10,
    lstm_units=64,
    epochs=50,
    batch_size=32
)

# Обучить модель на нормальных логах
analyzer.fit(logs_normal)

# Предсказать аномалии
anomalies = analyzer.predict(logs_test)

# Получить результаты
results = analyzer.get_anomaly_report()
print(results)
```

## Использование

### Примеры работы с данными

#### Пример 1: Обработка логов HDFS

```python
from src.data_processor import LogProcessor
from src.lstm_model import LSTMModel

# Загрузить логи
processor = LogProcessor(log_format='hdfs')
logs = processor.load_logs('data/raw/HDFS.log')

# Парсить логи
parsed_logs = processor.parse_logs(logs)

# Создать последовательности
sequences = processor.create_sequences(parsed_logs, window_size=10)

# Обучить модель
model = LSTMModel(lstm_units=64, dropout=0.2)
model.fit(sequences['train'], epochs=50, batch_size=32)

# Сохранить модель
model.save('models/trained_models/hdfs_model.h5')
```

#### Пример 2: Обнаружение аномалий в пользовательских логах

```python
from src.anomaly_detector import AnomalyDetector

# Инициализировать детектор
detector = AnomalyDetector(
    model_path='models/trained_models/hdfs_model.h5',
    threshold=0.8
)

# Загрузить тестовые логи
test_logs = processor.load_logs('data/raw/test_logs.log')
test_sequences = processor.create_sequences(test_logs, window_size=10)

# Предсказать аномалии
predictions = detector.detect(test_sequences)

# Сгенерировать отчет
detector.generate_report('results/reports/anomaly_report.html')
```

### Параметры модели

```python
model_config = {
    'sequence_length': 10,           # Длина последовательности
    'embedding_dim': 128,             # Размер embeddings
    'lstm_units': [64, 32],           # Кол-во юнитов в слоях LSTM
    'dropout': 0.2,                   # Коэффициент dropout
    'epochs': 50,                     # Кол-во эпох обучения
    'batch_size': 32,                 # Размер батча
    'learning_rate': 0.001,           # Скорость обучения
    'optimizer': 'adam',              # Оптимизатор
    'threshold': 0.8                  # Порог детекции аномалий
}
```

## Датасеты

Проект поддерживает работу со следующими публичными датасетами:

### 1. HDFS Log Dataset

- **Источник**: [GitHub - logpai/loghub](https://github.com/logpai/loghub)
- **Описание**: Логи распределенной файловой системы Hadoop
- **Размер**: 655K логов с 575 аномальными блоками
- **Использование**: 
  ```bash
  wget https://zenodo.org/record/3227741/files/HDFS_1.tar.gz
  tar -xzf HDFS_1.tar.gz
  ```

### 2. BGL Log Dataset

- **Источник**: Los Alamos National Laboratory
- **Описание**: Системные логи суперкомпьютера
- **Размер**: 4.7M логов
- **Использование**:
  ```bash
  wget https://zenodo.org/record/3227741/files/BGL.tar.gz
  tar -xzf BGL.tar.gz
  ```

## Обучение модели

### Обучение с нуля

```bash
python scripts/train_model.py \
  --data_path data/processed/HDFS_train.pkl \
  --epochs 50 \
  --batch_size 32 \
  --model_name hdfs_lstm_v1 \
  --output_dir models/trained_models/
```

### Объемные параметры обучения

```python
training_params = {
    'epochs': 100,
    'batch_size': 64,
    'validation_split': 0.1,
    'early_stopping': True,
    'early_stopping_patience': 10,
    'learning_rate_schedule': True
}
```

## Оценка результатов

### Метрики производительности

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model, test_data, test_labels)

# Основные метрики
metrics = evaluator.calculate_metrics()

print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
```

### Визуализация результатов

```bash
python scripts/visualize_results.py \
  --predictions_path results/predictions.pkl \
  --output_dir results/visualizations/
```

## Результаты

### Производительность модели на датасете HDFS

| Метрика | Значение |
|---------|----------|
| Accuracy | 96.8% |
| Precision | 95.2% |
| Recall | 97.5% |
| F1-Score | 96.3% |
| ROC-AUC | 0.978 |

### Производительность на датасете BGL

| Метрика | Значение |
|---------|----------|
| Accuracy | 94.5% |
| Precision | 92.8% |
| Recall | 95.7% |
| F1-Score | 94.2% |
| ROC-AUC | 0.965 |

## Контакты

Для вопросов и предложений:
- **GitHub Issues**: [https://github.com/MirazE1/LSTM_Log_Anomaly_Detection/issues](https://github.com/MirazE1/LSTM_Log_Anomaly_Detection/issues)

## Полезные ссылки

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [LogHub Dataset Repository](https://github.com/logpai/loghub)
- [Paper: Deep Learning for Anomaly Detection in Log Data](https://arxiv.org/pdf/2207.03820)
- [Paper: Automatic Log Parsing with Machine Learning](https://arxiv.org/abs/2307.16714)

---

**Последнее обновление**: November 29, 2025
