# OOD Detection

## Структура

```text
OOD_detection_actual/
├── data/
└── point_wise_detection/
    ├── partition_ood.py
    ├── partitioning_scheme_learning.py
    ├── metacharacteristics_taxonomy_light.md
    ├── partition_ood_experiments.ipynb
    ├── partition_ood_id_ood_experiments.ipynb
    ├── partitioning_scheme_experiment.ipynb
    ├── metacharacteristics_exclude_include_experiment.ipynb
    └── metacharacteristics_significance_experiment.ipynb
```

## Данные (`data/`)

Пары CSV для семи датасетов:


| Датасет       | ID                         | OOD                        |
| ------------- | -------------------------- | -------------------------- |
| Taxi          | `taxi_source.csv`          | `taxi_target.csv`          |
| Electricity   | `electricity_source.csv`   | `electricity_target.csv`   |
| Income        | `income_source.csv`        | `income_target.csv`        |
| MVx6          | `mv_x6_source.csv`         | `mv_x6_target.csv`         |
| Diabetes      | `diabites_source.csv`      | `diabites_target.csv`      |
| California    | `california_source.csv`    | `california_target.csv`    |
| ACS Accidents | `acs_accidents_source.csv` | `acs_accidents_target.csv` |


`source` — in-distribution, `target` — out-of-distribution.  
Загрузка идёт через `partition_ood.load_dataset()`: пути считаются от корня проекта, поэтому папка `data/` лежит рядом с `point_wise_detection/`.

## Ядро: `partition_ood.py`

Главный модуль проекта.

**Схемы разбиения**

- `QuantileBinningScheme` — квантильное разбиение по одному признаку.
- `KMeansScheme` — кластеризация KMeans по случайному подмножеству признаков.
- `DecisionTreeScheme` — дерево на синтетических метках; листья становятся группами.

`EnsemblePartitionOOD` собирает все три типа в один ensemble: quantiles по каждому признаку, несколько k-means схем, несколько tree-схем.

**Метахарактеристики**

Реестр — `ALL_META_FEATURES`. Рабочий список в экспериментах близок к light-таксономии: point-based расстояния и выбросы, marginal/feature-interaction статистики и информационные меры, базовые показатели поддержки ячейки (`log_count`, `density`, …).

**Обучение и оценка**

- `build_ood_dataset` — train/test split, fit разбиений, построение представления.
- `partition_fit_data='id'` или `'id+ood'` — на каких данных fit-ятся сами схемы (тестовые объекты в fit не попадают).
- `cross_validate_ood_classifiers` / `full_pipeline` — CV, балансировка классов, метрики (ROC-AUC, PR-AUC, F1 и др.), визуализации.



## Гиперпараметры схем: `partitioning_scheme_learning.py`

Надстройка над `partition_ood.py` для экспериментов, где важна **одна семья схем**, а не полный ensemble.

- `SingleSchemePartitionOOD` — только quantile, только k-means или только tree.
- Sweep по `n_bins`, `n_kmeans_clusters`, `tree_max_depth`, `n_tree_partitions`.
- Вариант с подбором гиперпараметров на validation-сплите.

Нужен для `partitioning_scheme_experiment.ipynb`.

## Таксономия: `metacharacteristics_taxonomy_light.md`

Компактная группировка метахарактеристик, на которую опирается significance-эксперимент:

1. **Point-based** — положение объекта относительно своей ячейки.
2. **Sample-based / Marginal-char** — статистика и энтропии по отдельным признакам внутри ячейки.
3. **Sample-based / Feature-interaction** — ковариации, корреляции, взаимная информация.
4. **Base** — размер/плотность ячейки, глубина листа, согласованность схем.

Полный расширенный taxonomy-файл в эту сборку не включён: в экспериментах используется light-версия.

## Эксперименты



### `partition_ood_experiments.ipynb`

Базовый прогон полного ensemble на датасетах.  
Смотрит, насколько хорошо ID/OOD разделяются при стандартных настройках схем и метахарактеристик. Точка отсчёта для остальных сравнений.

### `partition_ood_id_ood_experiments.ipynb`

Сравнивает два режима fit разбиений:

- только на ID;
- на ID + OOD (только train-сплиты).

Вопрос: меняется ли качество, если схемы «видят» OOD уже на этапе разбиения пространства. Прогнан на всех семи датасетах.

### `partitioning_scheme_experiment.ipynb`

Оценивает вклад отдельных семейств схем и их гиперпараметров:

- только quantile vs `n_bins`;
- только k-means vs `n_kmeans_clusters`;
- только tree vs глубина и число tree-схем;
- плюс вариант с обучаемыми гиперпараметрами.

Прогон прерывался на Taxi — результаты неполные.

### `metacharacteristics_exclude_include_experiment.ipynb`

Три представления объекта:

- только метахарактеристики;
- метахарактеристики + исходные признаки;
- только исходные признаки.

Показывает, дают ли meta-features прирост относительно сырых признаков. Частично прогнан (Taxi, Electricity, Income).

### `metacharacteristics_significance_experiment.ipynb`

Сравнивает **группы** метахарактеристик по light-таксономии, а не отдельные признаки по одному.

Для каждой группы (и для полного набора) считается качество классификации; итог — таблица и barchart по ROC-AUC. Позволяет понять, какие блоки метахарактеристик реально тянут качество.

## Как запускать

Рабочая директория — корень `OOD_detection_actual` (или `point_wise_detection`, если в ноутбуке настроен `sys.path` к родителю).

Минимальный пример:

```python
from point_wise_detection.partition_ood import load_dataset, full_pipeline

ID, OOD = load_dataset('Taxi')
results = full_pipeline(ID, OOD, n_splits=5, use_cv=True)
```

Конкретные сетки гиперпараметров и списки метахарактеристик задаются в соответствующих `.ipynb`.

