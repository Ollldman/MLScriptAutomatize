# Документация по работе модулей загрузки данных из различных источников.

## 1. load_from_static.py:
Модуль отвечает за реализацию получения pandas.DataFrame из статических файлов, сохраненных у пользователя на локальной машине, полученных как открытый поток, или получаемых асинхронно по API запросам.

Необходимо подготовить путь `filepath` до нужного файла в формате строки или иной удобный и доступный для pandas.read*** тип источника. (см. документацию pandas)

Представлены функции:
- `load_from_csv` для загрузки файлов из `*.csv`

и 

- `load_from_excel`, для загрузки файлов-excel таблиц.

Данные функции являются дублями функциональности от pandas.read***.

Для тестирования функций в этом модуле, путь к файлам таблиц берется из глобального конфигурационного файла `.env`

## 2. async_load_from_db.py
Модуль отвечает за реализацию получения pandas.DataFrame из реляционных баз данных с использованием SQL-запросов, сервер которых реализован у пользователя на локальной машине.

В данный момент доступна функциональность только для PostgreSQL.

- `class AsyncSQLRunner()`

В модуле представлена реализация объекта SQLRunner отвечающего за асинхронное взаимодействие с базой данных пользователя PostgreSQL по передаваемой строке соединения. (Для тестирования работы методов, используется строка подключения из .env файла проекта, указанная в settings.py).

- `def async_sql_to_df(...)`

Дополнительно, для упрощения работы, представлена функция async_sql_to_df(...) которая возвращает pandas.DataFrame по переданной строке SQL-запроса, строке соединения и дополнительных необходимых параметрах, которые используются в библиотечном методе pandas.read_sql_query(...).

Реализация тесно связана с ограничением на выполнение только SELECT -запросов к базе данных с целью получения релевантных табличных данных и с целью обеспечения безопасности базы данных от инъекций и ошибочных запросов.

Необходимая функциональность в будущем:
- Валидация имён таблиц/столбцов из allowlist,
- Ограничение по времени выполнения запроса,
- Ограничение по объёму возвращаемых данных,
- Аудит логов: кто, когда, какой SELECT выполнил,
- Интеграция с Pydantic для валидации структуры результата.
     
## 3. load_from_api.py:

- `def load_from_kaggle(...)`

Загружает датасет с Kaggle через kagglehub с использованием PANDAS адаптера.

    :param dataset_id: Например, "username/dataset_name"
    :param filename: Имя файла внутри датасета (например, "winemag-data-130k-v2.csv"). 
    :param kwargs: Дополнительные аргументы, передаваемые в pd.read_csv (если используется fallback)
    :return: pd.DataFrame или None в случае ошибки

Особое внимание следует уделить API токену, полученному из вашего аккаунта на kaggle. Он подхватывается либо из файла .env,  либо из текущих переменных окружения.

- `def load_from_huggingface(...)`

Загружает датасет с Hugging Face Datasets.

    :param dataset_id: Например, "imdb", "glue", "rajpurkar/squad", и т.д.
    :param config_name: Конфиг (например, "mrpc" для glue)
    :param split: "train", "test", "validation"
    :return: pd.DataFrame или None

- `def load_from_uci(...)`

Загружает датасет с UCI Machine Learning Repository.

    :dataset_id: ID набора данных на uci
    :return: pd.DataFrame или None

- `def load_from_sklearn(...)`

Загружает встроенный датасет из sklearn.datasets.

    :param name: Имя функции, например, "load_iris", "load_boston", "fetch_california_housing"
    :return: pd.DataFrame или None

Реализован основной модуль для загрузки из любого источника, который вызывает и обрабатывает результаты работы из ранее указанных источников.

- `def load_dataset(...)`

Универсальная функция загрузки датасетов.

    Поддерживаемые источники:
      - 'kaggle': dataset_id, filename
      - 'huggingface': dataset_id, config_name, split
      - 'uci': url
      - 'sklearn': name

    Примеры:
      load_dataset('kaggle', dataset_id='zynicide/wine-reviews', filename='winemag-data_first150k.csv')
      load_dataset('huggingface', dataset_id='imdb', split='test')
      load_dataset('uci', url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
      load_dataset('sklearn', name='load_iris')
Загрузка данных выполняется явным указанием источника в аргументы этого модуля, и аргументов, которые передадутся в указанны модуль, аргументы после источника должны соответствовать документации функции-источника, указанной выше.
Обработка идет по ограниченному списку запрашиваемых методов:
loaders = {
        'kaggle': load_from_kaggle,
        'huggingface': load_from_huggingface,
        'uci': load_from_uci,
        'sklearn': load_from_sklearn,
    }