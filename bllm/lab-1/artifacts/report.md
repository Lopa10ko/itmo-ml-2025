# Поиск товаров с помощью LangChain и SQLite

## Цель работы и порядок запуска

Разработать интерфейс к каталогу товаров, который принимает запрос на естественном
языке, извлекает параметры с помощью LLM, проверяет их по схеме Pydantic и выполняет
поиск в SQLite.

Инструкция по запуску приведена в [README](../README.md). Полные результаты
экспериментов сохранены в [results.json](results.json), а последовательность событий
каждого запроса — в [events.jsonl](events.jsonl). Для каждого эксперимента указан
источник ответов. В режиме `offline` используются заранее заданные ответы модели.

Режим выполнения: `live`. Начало: 2026-11-09T13:10:06.359134+00:00. Завершение: 2026-11-09T13:10:27.563004+00:00. Время указано в UTC.

| Параметр | Значение |
| --- | --- |
| Проверка целостности SQLite | ok |
| Версии библиотек | {"langchain": "1.4.9", "langchain-core": "1.7.6", "langchain-openai": "1.7.4", "pydantic": "2.14.2"} |
| Модель | qwen3.8-27b |
| Адрес API | http://10.32.2.11:35000/v1 |
| Температура генерации | 0.0 |
| Лимит выходных токенов | 2048 |
| Максимальное число попыток | 3 |
| Параметры генерации каталога | {"products_per_category": 30, "seed": 42} |

### Каталог эксперимента

| ID | Название | Категория | Цена, руб. |
| --- | --- | --- | --- |
| 1 | Entry-level laptop | laptop | 35000 |
| 2 | Compact laptop | laptop | 50000 |
| 3 | Office laptop | laptop | 70000 |
| 4 | Gaming laptop | laptop | 80000 |
| 5 | Professional laptop | laptop | 120000 |
| 6 | Entry-level smartphone | smartphone | 15000 |
| 7 | Compact smartphone | smartphone | 30000 |
| 8 | Smartphone with a large battery | smartphone | 50000 |
| 9 | Camera-focused smartphone | smartphone | 50000 |
| 10 | Flagship smartphone | smartphone | 100000 |
| 11 | Entry-level tablet | tablet | 12000 |
| 12 | Student tablet | tablet | 25000 |
| 13 | Drawing tablet | tablet | 40000 |
| 14 | Productivity tablet | tablet | 60000 |
| 15 | Professional tablet | tablet | 90000 |
| 16 | Entry-level laptop variant 01 | laptop | 35000 |
| 17 | Entry-level smartphone variant 01 | smartphone | 15000 |
| 18 | Entry-level tablet variant 01 | tablet | 12000 |
| 19 | Compact laptop variant 02 | laptop | 35001 |
| 20 | Compact smartphone variant 02 | smartphone | 15001 |
| 21 | Student tablet variant 02 | tablet | 12001 |
| 22 | Office laptop variant 03 | laptop | 49999 |
| 23 | Smartphone with a large battery variant 03 | smartphone | 29999 |
| 24 | Drawing tablet variant 03 | tablet | 24999 |
| 25 | Gaming laptop variant 04 | laptop | 50000 |
| 26 | Camera-focused smartphone variant 04 | smartphone | 30000 |
| 27 | Productivity tablet variant 04 | tablet | 25000 |
| 28 | Professional laptop variant 05 | laptop | 50001 |
| 29 | Flagship smartphone variant 05 | smartphone | 30001 |
| 30 | Professional tablet variant 05 | tablet | 25001 |
| 31 | Entry-level laptop variant 06 | laptop | 69999 |
| 32 | Entry-level smartphone variant 06 | smartphone | 49999 |
| 33 | Entry-level tablet variant 06 | tablet | 39999 |
| 34 | Compact laptop variant 07 | laptop | 70000 |
| 35 | Compact smartphone variant 07 | smartphone | 50000 |
| 36 | Student tablet variant 07 | tablet | 40000 |
| 37 | Office laptop variant 08 | laptop | 70001 |
| 38 | Smartphone with a large battery variant 08 | smartphone | 50001 |
| 39 | Drawing tablet variant 08 | tablet | 40001 |
| 40 | Gaming laptop variant 09 | laptop | 79999 |
| 41 | Camera-focused smartphone variant 09 | smartphone | 99999 |
| 42 | Productivity tablet variant 09 | tablet | 59999 |
| 43 | Professional laptop variant 10 | laptop | 80000 |
| 44 | Flagship smartphone variant 10 | smartphone | 100000 |
| 45 | Professional tablet variant 10 | tablet | 60000 |
| 46 | Entry-level laptop variant 11 | laptop | 80001 |
| 47 | Entry-level smartphone variant 11 | smartphone | 79726 |
| 48 | Entry-level tablet variant 11 | tablet | 60001 |
| 49 | Compact laptop variant 12 | laptop | 119999 |
| 50 | Compact smartphone variant 12 | smartphone | 48850 |
| 51 | Student tablet variant 12 | tablet | 89999 |
| 52 | Office laptop variant 13 | laptop | 120000 |
| 53 | Smartphone with a large battery variant 13 | smartphone | 54951 |
| 54 | Drawing tablet variant 13 | tablet | 90000 |
| 55 | Gaming laptop variant 14 | laptop | 37274 |
| 56 | Camera-focused smartphone variant 14 | smartphone | 71759 |
| 57 | Productivity tablet variant 14 | tablet | 60887 |
| 58 | Professional laptop variant 15 | laptop | 60249 |
| 59 | Flagship smartphone variant 15 | smartphone | 60054 |
| 60 | Professional tablet variant 15 | tablet | 31246 |
| 61 | Entry-level laptop variant 16 | laptop | 100428 |
| 62 | Entry-level smartphone variant 16 | smartphone | 69024 |
| 63 | Entry-level tablet variant 16 | tablet | 55364 |
| 64 | Compact laptop variant 17 | laptop | 107234 |
| 65 | Compact smartphone variant 17 | smartphone | 38544 |
| 66 | Student tablet variant 17 | tablet | 30293 |
| 67 | Office laptop variant 18 | laptop | 93127 |
| 68 | Smartphone with a large battery variant 18 | smartphone | 59094 |
| 69 | Drawing tablet variant 18 | tablet | 73174 |
| 70 | Gaming laptop variant 19 | laptop | 86760 |
| 71 | Camera-focused smartphone variant 19 | smartphone | 73132 |
| 72 | Productivity tablet variant 19 | tablet | 14970 |
| 73 | Professional laptop variant 20 | laptop | 102439 |
| 74 | Flagship smartphone variant 20 | smartphone | 50377 |
| 75 | Professional tablet variant 20 | tablet | 20958 |
| 76 | Entry-level laptop variant 21 | laptop | 65704 |
| 77 | Entry-level smartphone variant 21 | smartphone | 99916 |
| 78 | Entry-level tablet variant 21 | tablet | 54742 |
| 79 | Compact laptop variant 22 | laptop | 93726 |
| 80 | Compact smartphone variant 22 | smartphone | 19856 |
| 81 | Student tablet variant 22 | tablet | 43488 |
| 82 | Office laptop variant 23 | laptop | 100756 |
| 83 | Smartphone with a large battery variant 23 | smartphone | 87476 |
| 84 | Drawing tablet variant 23 | tablet | 22493 |
| 85 | Gaming laptop variant 24 | laptop | 66749 |
| 86 | Camera-focused smartphone variant 24 | smartphone | 19278 |
| 87 | Productivity tablet variant 24 | tablet | 87491 |
| 88 | Professional laptop variant 25 | laptop | 45262 |
| 89 | Flagship smartphone variant 25 | smartphone | 26403 |
| 90 | Professional tablet variant 25 | tablet | 27499 |

## Архитектура и правила обработки запросов

В основе приложения лежат неизменяемые модели данных `ProductQuery` и `Product`,
а также интерфейс репозитория. Прикладной сценарий `SearchProducts` получает
репозиторий через конструктор. Модули `persistence`, `llm` и `review` отвечают
соответственно за доступ к SQLite, работу с моделью и проведение экспериментов.
Зависимости собираются в `bootstrap` и `entrypoints`.

Поле `category` принимает значения `laptop`, `smartphone`, `tablet` и `other`.
Поле `max_price` содержит целое неотрицательное число рублей или `null`, если
ограничения цены нет. Поле `sort_by` задаёт порядок сортировки; по умолчанию
используется `price_asc`. Полная схема с описаниями полей, ограничениями и значениями
по умолчанию сохранена в `results.json`.

Метод `PydanticOutputParser.get_format_instructions()` формирует инструкции,
которые включаются в системное сообщение `system`. Запрос пользователя передаётся
в сообщении `human`. Компоненты LCEL соединяются оператором `|`. Парсер возвращает
проверенный объект `ProductQuery`, который затем сериализуется в JSON.

Первые пять запросов взяты из задания. В двух дополнительных примерах приняты
явные допущения: слово «недорогой» задаёт сортировку по возрастанию цены без
числового бюджета, а «до полтинника» означает не более 50 000 рублей. Поскольку
цены заданы в целых рублях, «дешевле N» преобразуется в предел N−1, а «до N»
включает саму границу N.

SQL-запрос составляет приложение. Категория и цена передаются через параметры,
а направление сортировки выбирается из разрешённого списка. При равных ценах
товары упорядочиваются по `id`. Для каждого поиска открывается отдельное соединение
в режиме чтения, которое закрывается после получения результата. Новая база
становится доступна только после успешного создания таблицы и записи данных.
Существующие файлы при инициализации не заменяются.

Синтетический каталог создаётся на основе `fixtures/products.json`. Его размер
и начальное значение генератора случайных чисел задаются в `.env`. Генератор
сохраняет исходные примеры и добавляет товары с равными ценами, ценами по обе стороны
от границ фильтрации и случайными ценами внутри заданного диапазона.

При извлечении параметров из JSON текст ответа разбирает `PydanticOutputParser`.
При вызове инструмента схема `args_schema=ProductQuery` передаётся модели через
`bind_tools`. Приложение проверяет имя инструмента, число вызовов и аргументы,
а затем выполняет функцию, оформленную через `@tool`. Найденные товары возвращаются
приложению без повторной передачи модели. Если вызова нет или его аргументы
невалидны, поиск не выполняется. Для категории `other` возвращается сообщение
о неподдерживаемой категории.

Механизм `with_retry` повторяет генерацию и разбор ответа: допускается не более
трёх попыток, включая первую. Поиск в SQLite выполняется после успешной валидации
и не входит в повторяемую часть цепочки. Встроенные сетевые повторы клиента
отключены. Журнал содержит `request_id`, номера попыток, сообщения промпта,
результаты разбора и валидации, запуск поиска и число найденных товаров.
Подтверждение поиска без ограничения цены задаётся в `.env` до выполнения поиска.

Режим `enable_thinking=True` используется только в эксперименте задания 2.4.
Сравниваются ответы на одни и те же запросы. Время и число токенов относятся
к одному запуску и не позволяют сделать статистический вывод о быстродействии.
Дополнительный диагностический запрос сохраняет исходное сообщение сервера
в записи `raw_reasoning_probe`; его время не включается в сравнение. Содержание
рассуждения не входит в схему `ProductQuery`. Клиент `ChatOpenAI` может не сохранить
нестандартное поле `reasoning_content`, поэтому для анализа используется и исходный
ответ сервера. JSON-парсеру передаётся содержимое поля `content`.

Для уточняющих запросов реализованы два способа передачи контекста. Первый хранит
последний валидный `ProductQuery` в Python и передаёт его вместе с новой репликой.
Второй передаёт всю историю сообщений пользователя и модели. Невалидный ответ
не заменяет сохранённые параметры. Оба способа проверяются на одинаковых диалогах;
число входных токенов берётся из `usage_metadata`.

## Базовая LCEL-цепочка

Источник ответов: университетская модель.

| Запрос | Ответ |
| --- | --- |
| Найди ноутбуки дешевле 70 000 рублей | Ноутбуки |
| Покажи самые дорогие смартфоны | Смартфоны |
| Хочу посмотреть планшеты | Планшеты |

Прямой вызов модели вернул `AIMessage`. После парсера получен `TextAccessor`; проверка `isinstance(result, str)` подтверждает, что это строка.

Дополнительный запуск с `language=English`: Laptops

## Структурированный ответ: проверка полей

Источник ответов: университетская модель.

| ID | Запрос | Полученные параметры | Проверка полей |
| --- | --- | --- | --- |
| Q1 | Найди ноутбуки дешевле 70 000 рублей | {"category": "laptop", "max_price": 69999, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q2 | Покажи самые дорогие смартфоны | {"category": "smartphone", "max_price": null, "sort_by": "price_desc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q3 | Хочу посмотреть планшеты | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q4 | Покажи ноутбуки до 50 тысяч | {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q5 | Найди холодильники | {"category": "other", "max_price": null, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q6 | Нужен недорогой ноутбук | {"category": "laptop", "max_price": null, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |
| Q7 | Покажи телефоны до полтинника | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | {"category": true, "max_price": true, "sort_by": true} |

«Недорогой» трактуется как сортировка по возрастанию цены без числового ограничения.

В контексте покупки электроники «до полтинника» здесь означает не более 50 000 рублей. Это допущение задано явно.

## Поиск по SQLite

Источник ответов: университетская модель.

| Запрос пользователя | Структурированные параметры | Найденные товары | Корректность |
| --- | --- | --- | --- |
| Найди ноутбуки дешевле 70 000 рублей | {"category": "laptop", "max_price": 69999, "sort_by": "price_asc"} | ["1: Entry-level laptop (35000)", "16: Entry-level laptop variant 01 (35000)", "19: Compact laptop variant 02 (35001)", "55: Gaming laptop variant 14 (37274)", "88: Professional laptop variant 25 (45262)", "22: Office laptop variant 03 (49999)", "2: Compact laptop (50000)", "25: Gaming laptop variant 04 (50000)", "28: Professional laptop variant 05 (50001)", "58: Professional laptop variant 15 (60249)", "76: Entry-level laptop variant 21 (65704)", "85: Gaming laptop variant 24 (66749)", "31: Entry-level laptop variant 06 (69999)"] | Да |
| Покажи самые дорогие смартфоны | {"category": "smartphone", "max_price": null, "sort_by": "price_desc"} | ["10: Flagship smartphone (100000)", "44: Flagship smartphone variant 10 (100000)", "41: Camera-focused smartphone variant 09 (99999)", "77: Entry-level smartphone variant 21 (99916)", "83: Smartphone with a large battery variant 23 (87476)", "47: Entry-level smartphone variant 11 (79726)", "71: Camera-focused smartphone variant 19 (73132)", "56: Camera-focused smartphone variant 14 (71759)", "62: Entry-level smartphone variant 16 (69024)", "59: Flagship smartphone variant 15 (60054)", "68: Smartphone with a large battery variant 18 (59094)", "53: Smartphone with a large battery variant 13 (54951)", "74: Flagship smartphone variant 20 (50377)", "38: Smartphone with a large battery variant 08 (50001)", "8: Smartphone with a large battery (50000)", "9: Camera-focused smartphone (50000)", "35: Compact smartphone variant 07 (50000)", "32: Entry-level smartphone variant 06 (49999)", "50: Compact smartphone variant 12 (48850)", "65: Compact smartphone variant 17 (38544)", "29: Flagship smartphone variant 05 (30001)", "7: Compact smartphone (30000)", "26: Camera-focused smartphone variant 04 (30000)", "23: Smartphone with a large battery variant 03 (29999)", "89: Flagship smartphone variant 25 (26403)", "80: Compact smartphone variant 22 (19856)", "86: Camera-focused smartphone variant 24 (19278)", "20: Compact smartphone variant 02 (15001)", "6: Entry-level smartphone (15000)", "17: Entry-level smartphone variant 01 (15000)"] | Да |
| Хочу посмотреть планшеты | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | ["11: Entry-level tablet (12000)", "18: Entry-level tablet variant 01 (12000)", "21: Student tablet variant 02 (12001)", "72: Productivity tablet variant 19 (14970)", "75: Professional tablet variant 20 (20958)", "84: Drawing tablet variant 23 (22493)", "24: Drawing tablet variant 03 (24999)", "12: Student tablet (25000)", "27: Productivity tablet variant 04 (25000)", "30: Professional tablet variant 05 (25001)", "90: Professional tablet variant 25 (27499)", "66: Student tablet variant 17 (30293)", "60: Professional tablet variant 15 (31246)", "33: Entry-level tablet variant 06 (39999)", "13: Drawing tablet (40000)", "36: Student tablet variant 07 (40000)", "39: Drawing tablet variant 08 (40001)", "81: Student tablet variant 22 (43488)", "78: Entry-level tablet variant 21 (54742)", "63: Entry-level tablet variant 16 (55364)", "42: Productivity tablet variant 09 (59999)", "14: Productivity tablet (60000)", "45: Professional tablet variant 10 (60000)", "48: Entry-level tablet variant 11 (60001)", "57: Productivity tablet variant 14 (60887)", "69: Drawing tablet variant 18 (73174)", "87: Productivity tablet variant 24 (87491)", "51: Student tablet variant 12 (89999)", "15: Professional tablet (90000)", "54: Drawing tablet variant 13 (90000)"] | Да |
| Покажи ноутбуки до 50 тысяч | {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"} | ["1: Entry-level laptop (35000)", "16: Entry-level laptop variant 01 (35000)", "19: Compact laptop variant 02 (35001)", "55: Gaming laptop variant 14 (37274)", "88: Professional laptop variant 25 (45262)", "22: Office laptop variant 03 (49999)", "2: Compact laptop (50000)", "25: Gaming laptop variant 04 (50000)"] | Да |
| Найди холодильники | {"category": "other", "max_price": null, "sort_by": "price_asc"} | [] | Да |
| Нужен недорогой ноутбук | {"category": "laptop", "max_price": null, "sort_by": "price_asc"} | ["1: Entry-level laptop (35000)", "16: Entry-level laptop variant 01 (35000)", "19: Compact laptop variant 02 (35001)", "55: Gaming laptop variant 14 (37274)", "88: Professional laptop variant 25 (45262)", "22: Office laptop variant 03 (49999)", "2: Compact laptop (50000)", "25: Gaming laptop variant 04 (50000)", "28: Professional laptop variant 05 (50001)", "58: Professional laptop variant 15 (60249)", "76: Entry-level laptop variant 21 (65704)", "85: Gaming laptop variant 24 (66749)", "31: Entry-level laptop variant 06 (69999)", "3: Office laptop (70000)", "34: Compact laptop variant 07 (70000)", "37: Office laptop variant 08 (70001)", "40: Gaming laptop variant 09 (79999)", "4: Gaming laptop (80000)", "43: Professional laptop variant 10 (80000)", "46: Entry-level laptop variant 11 (80001)", "70: Gaming laptop variant 19 (86760)", "67: Office laptop variant 18 (93127)", "79: Compact laptop variant 22 (93726)", "61: Entry-level laptop variant 16 (100428)", "82: Office laptop variant 23 (100756)", "73: Professional laptop variant 20 (102439)", "64: Compact laptop variant 17 (107234)", "49: Compact laptop variant 12 (119999)", "5: Professional laptop (120000)", "52: Office laptop variant 13 (120000)"] | Да |
| Покажи телефоны до полтинника | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | ["6: Entry-level smartphone (15000)", "17: Entry-level smartphone variant 01 (15000)", "20: Compact smartphone variant 02 (15001)", "86: Camera-focused smartphone variant 24 (19278)", "80: Compact smartphone variant 22 (19856)", "89: Flagship smartphone variant 25 (26403)", "23: Smartphone with a large battery variant 03 (29999)", "7: Compact smartphone (30000)", "26: Camera-focused smartphone variant 04 (30000)", "29: Flagship smartphone variant 05 (30001)", "65: Compact smartphone variant 17 (38544)", "50: Compact smartphone variant 12 (48850)", "32: Entry-level smartphone variant 06 (49999)", "8: Smartphone with a large battery (50000)", "9: Camera-focused smartphone (50000)", "35: Compact smartphone variant 07 (50000)"] | Да |
| Покажи планшеты до 0 рублей | {"category": "tablet", "max_price": 0, "sort_by": "price_asc"} | [] | Да |
| Покажи ноутбуки до 80 тысяч, сначала дорогие | {"category": "laptop", "max_price": 80000, "sort_by": "price_desc"} | ["4: Gaming laptop (80000)", "43: Professional laptop variant 10 (80000)", "40: Gaming laptop variant 09 (79999)", "37: Office laptop variant 08 (70001)", "3: Office laptop (70000)", "34: Compact laptop variant 07 (70000)", "31: Entry-level laptop variant 06 (69999)", "85: Gaming laptop variant 24 (66749)", "76: Entry-level laptop variant 21 (65704)", "58: Professional laptop variant 15 (60249)", "28: Professional laptop variant 05 (50001)", "2: Compact laptop (50000)", "25: Gaming laptop variant 04 (50000)", "22: Office laptop variant 03 (49999)", "88: Professional laptop variant 25 (45262)", "55: Gaming laptop variant 14 (37274)", "19: Compact laptop variant 02 (35001)", "1: Entry-level laptop (35000)", "16: Entry-level laptop variant 01 (35000)"] | Да |
| Покажи самые дешевые смартфоны до 50 000 рублей | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | ["6: Entry-level smartphone (15000)", "17: Entry-level smartphone variant 01 (15000)", "20: Compact smartphone variant 02 (15001)", "86: Camera-focused smartphone variant 24 (19278)", "80: Compact smartphone variant 22 (19856)", "89: Flagship smartphone variant 25 (26403)", "23: Smartphone with a large battery variant 03 (29999)", "7: Compact smartphone (30000)", "26: Camera-focused smartphone variant 04 (30000)", "29: Flagship smartphone variant 05 (30001)", "65: Compact smartphone variant 17 (38544)", "50: Compact smartphone variant 12 (48850)", "32: Entry-level smartphone variant 06 (49999)", "8: Smartphone with a large battery (50000)", "9: Camera-focused smartphone (50000)", "35: Compact smartphone variant 07 (50000)"] | Да |

## Поиск как инструмент

Источник ответов: университетская модель.

| ID | Текст ответа | Вызовы `tool_calls` | Параметры | Статус | Верно |
| --- | --- | --- | --- | --- | --- |
| Q1 | — | [{"name": "search_products", "args": {"category": "laptop", "max_price": 69999, "sort_by": "price_asc"}, "id": "chatcmpl-tool-b2b0afbcc0fc477c", "type": "tool_call"}] | {"category": "laptop", "max_price": 69999, "sort_by": "price_asc"} | found | Да |
| Q2 | — | [{"name": "search_products", "args": {"category": "smartphone", "sort_by": "price_desc"}, "id": "chatcmpl-tool-b2a4f4ae186980de", "type": "tool_call"}] | {"category": "smartphone", "max_price": null, "sort_by": "price_desc"} | found | Да |
| Q3 | — | [{"name": "search_products", "args": {"category": "tablet"}, "id": "chatcmpl-tool-8c8a56a2b82c54c4", "type": "tool_call"}] | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | found | Да |
| Q4 | — | [{"name": "search_products", "args": {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"}, "id": "chatcmpl-tool-bc6a7d34495b2330", "type": "tool_call"}] | {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"} | found | Да |
| Q5 | — | [{"name": "search_products", "args": {"category": "other"}, "id": "chatcmpl-tool-97c47edc70bbd77c", "type": "tool_call"}] | {"category": "other", "max_price": null, "sort_by": "price_asc"} | unsupported_category | Да |
| Q6 | — | [{"name": "search_products", "args": {"category": "laptop", "sort_by": "price_asc"}, "id": "chatcmpl-tool-a53641f8a780eeb1", "type": "tool_call"}] | {"category": "laptop", "max_price": null, "sort_by": "price_asc"} | found | Да |
| Q7 | — | [{"name": "search_products", "args": {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"}, "id": "chatcmpl-tool-b239e5ad433d58f3", "type": "tool_call"}] | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | found | Да |

Найденные товары возвращены приложению. После поиска модель повторно не вызывается.

## Сравнение режимов размышления

Источник ответов: университетская модель.

| ID | Режим размышления | Параметры | Время, с | Выходные токены | Попытки | Результат разбора |
| --- | --- | --- | --- | --- | --- | --- |
| Q6 | Нет | {"category": "laptop", "max_price": null, "sort_by": "price_asc"} | 0.37347849970683455 | 36 | 1 | validated |
| Q7 | Нет | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | 0.394270250108093 | 41 | 1 | validated |
| Q6 | Да | {"category": "laptop", "max_price": null, "sort_by": "price_asc"} | 0.8486133338883519 | 104 | 1 | validated |
| Q7 | Да | {"category": "smartphone", "max_price": 50000, "sort_by": "price_asc"} | 1.0612365831620991 | 121 | 1 | validated |

Исходное сообщение сервера сохранено в записи `raw_reasoning_probe` этого раздела в `results.json`. Время диагностического запроса не включено в сравнение.

Поля рассуждения с непустым значением в исходном ответе: reasoning_content. Парсеру передаётся поле `content`. Полное диагностическое сообщение сохранено в `results.json`.

## Уточняющие запросы: параметры и история диалога

Источник ответов: университетская модель.

| Диалог | Контекст | Ход | Реплика | Ожидаемые параметры | Полученные параметры | Верно | Входные токены |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | state | 1 | Покажи ноутбуки до 80 тысяч | {"category": "laptop", "max_price": 80000, "sort_by": "price_asc"} | {"category": "laptop", "max_price": 80000, "sort_by": "price_asc"} | Да | 544 |
| A | state | 2 | А сначала дорогие? | {"category": "laptop", "max_price": 80000, "sort_by": "price_desc"} | {"category": "laptop", "max_price": 80000, "sort_by": "price_desc"} | Да | 564 |
| A | state | 3 | А что есть среди смартфонов? | {"category": "smartphone", "max_price": 80000, "sort_by": "price_desc"} | {"category": "smartphone", "max_price": 80000, "sort_by": "price_desc"} | Да | 567 |
| B | state | 1 | Покажи планшеты до 40 тысяч | {"category": "tablet", "max_price": 40000, "sort_by": "price_asc"} | {"category": "tablet", "max_price": 40000, "sort_by": "price_asc"} | Да | 545 |
| B | state | 2 | Теперь без ограничения цены | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | Да | 563 |
| B | state | 3 | Тогда сначала дорогие | {"category": "tablet", "max_price": null, "sort_by": "price_desc"} | {"category": "tablet", "max_price": null, "sort_by": "price_desc"} | Да | 558 |
| A | history | 1 | Покажи ноутбуки до 80 тысяч | {"category": "laptop", "max_price": 80000, "sort_by": "price_asc"} | {"category": "laptop", "max_price": 80000, "sort_by": "price_asc"} | Да | 544 |
| A | history | 2 | А сначала дорогие? | {"category": "laptop", "max_price": 80000, "sort_by": "price_desc"} | {"category": "laptop", "max_price": 80000, "sort_by": "price_desc"} | Да | 584 |
| A | history | 3 | А что есть среди смартфонов? | {"category": "smartphone", "max_price": 80000, "sort_by": "price_desc"} | {"category": "smartphone", "max_price": 80000, "sort_by": "price_desc"} | Да | 626 |
| B | history | 1 | Покажи планшеты до 40 тысяч | {"category": "tablet", "max_price": 40000, "sort_by": "price_asc"} | {"category": "tablet", "max_price": 40000, "sort_by": "price_asc"} | Да | 545 |
| B | history | 2 | Теперь без ограничения цены | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | {"category": "tablet", "max_price": null, "sort_by": "price_asc"} | Да | 584 |
| B | history | 3 | Тогда сначала дорогие | {"category": "tablet", "max_price": null, "sort_by": "price_desc"} | {"category": "tablet", "max_price": null, "sort_by": "price_desc"} | Да | 619 |

## Ответы без доступа к каталогу

Источник ответов: университетская модель.

| Запрос | Ответ модели без доступа к базе | ID найденных товаров |
| --- | --- | --- |
| Найди ноутбуки дешевле 70 000 рублей | Я не имею доступа к вашему конкретному каталогу товаров, поэтому не могу предоставить список ноутбуков дешевле 70 000 рублей.<br><br>Пожалуйста, уточните, какой именно каталог или магазин вы имеете в виду, или предоставьте данные о товарах, чтобы я мог помочь с поиском. | [1, 16, 19, 55, 88, 22, 2, 25, 28, 58, 76, 85, 31] |
| Покажи самые дорогие смартфоны | Я не имею доступа к вашему конкретному каталогу товаров, поэтому не могу показать список самых дорогих смартфонов именно из него.<br><br>Пожалуйста, уточните, какой именно каталог или магазин вы имеете в виду, или предоставьте данные, чтобы я мог помочь с анализом. | [10, 44, 41, 77, 83, 47, 71, 56, 62, 59, 68, 53, 74, 38, 8, 9, 35, 32, 50, 65, 29, 7, 26, 23, 89, 80, 86, 20, 6, 17] |
| Покажи ноутбуки до 50 тысяч | У меня нет доступа к вашему конкретному каталогу товаров, поэтому я не могу показать актуальный список ноутбуков до 50 000 рублей.<br><br>Пожалуйста, уточните, какой именно каталог или магазин вы имеете в виду, или предоставьте данные, чтобы я мог помочь с поиском. | [1, 16, 19, 55, 88, 22, 2, 25] |

## Ошибки, повторные попытки и подтверждение

Источник ответов: заранее заданные ответы модели.

| Сценарий | Проверяемые данные | Фактический результат | Верно |
| --- | --- | --- | --- |
| invalid_json | not-json | Исключение: OutputParserException; причина: JSONDecodeError; этап: json_parsing | Да |
| invalid_category | {"category":"refrigerator"} | Исключение: OutputParserException; причина: ValidationError; этап: validation | Да |
| string_price | {"category":"laptop","max_price":"50000"} | Исключение: OutputParserException; причина: ValidationError; этап: validation | Да |
| negative_price | {"category":"laptop","max_price":-1} | Исключение: OutputParserException; причина: ValidationError; этап: validation | Да |
| string_none | {"category":"laptop","max_price":"None"} | Исключение: OutputParserException; причина: ValidationError; этап: validation | Да |
| without_retry | Число попыток модели и вызовов поиска | Модель: 1; поиск: 0; статус: parse_error | Да |
| recovered | Число попыток модели и вызовов поиска | Модель: 2; поиск: 1; статус: found | Да |
| exhausted | Число попыток модели и вызовов поиска | Модель: 3; поиск: 0; статус: parse_error | Да |
| immediate_success | Число попыток модели и вызовов поиска | Модель: 1; поиск: 1; статус: found | Да |
| confirmation/False | Число попыток модели и вызовов поиска | Модель: 1; поиск: 0; статус: confirmation_required | Да |
| confirmation/True | Число попыток модели и вызовов поиска | Модель: 1; поиск: 1; статус: found | Да |
| absent_tool | Число попыток модели и вызовов поиска | Модель: 1; поиск: 0; статус: invalid_tool | Да |

Ответы модели в этом разделе заданы заранее. Используются те же парсер, механизм повторных попыток LCEL и репозиторий SQLite, что и в основном пайплайне. События запроса можно найти в `events.jsonl` по его `request_id`.

## Фрагмент журнала: успешный повтор после ошибки

```json
[
  {
    "timestamp": "2026-11-09T13:10:27.533665+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "input_received",
    "text": "Покажи ноутбуки до 50 тысяч"
  },
  {
    "timestamp": "2026-11-09T13:10:27.535148+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "prompt_formed",
    "attempt": 1
  },
  {
    "timestamp": "2026-11-09T13:10:27.535273+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "llm_started",
    "attempt": 1
  },
  {
    "timestamp": "2026-11-09T13:10:27.535862+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "llm_completed",
    "attempt": 1,
    "elapsed_seconds": 0.0004947497509419918,
    "usage": null
  },
  {
    "timestamp": "2026-11-09T13:10:27.536211+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "parse_failed",
    "attempt": 1,
    "error_type": "OutputParserException",
    "cause_type": "JSONDecodeError",
    "stage": "json_parsing"
  },
  {
    "timestamp": "2026-11-09T13:10:27.536302+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "retry_scheduled",
    "next_attempt": 2
  },
  {
    "timestamp": "2026-11-09T13:10:27.536910+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "prompt_formed",
    "attempt": 2
  },
  {
    "timestamp": "2026-11-09T13:10:27.537018+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "llm_started",
    "attempt": 2
  },
  {
    "timestamp": "2026-11-09T13:10:27.537189+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "llm_completed",
    "attempt": 2,
    "elapsed_seconds": 9.624985978007317e-05,
    "usage": null
  },
  {
    "timestamp": "2026-11-09T13:10:27.537443+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "parsed",
    "attempt": 2
  },
  {
    "timestamp": "2026-11-09T13:10:27.537524+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "validated",
    "parameters": {
      "category": "laptop",
      "max_price": 50000,
      "sort_by": "price_asc"
    }
  },
  {
    "timestamp": "2026-11-09T13:10:27.537733+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "search_started",
    "parameters": {
      "category": "laptop",
      "max_price": 50000,
      "sort_by": "price_asc"
    }
  },
  {
    "timestamp": "2026-11-09T13:10:27.538330+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "search_completed",
    "count": 8
  },
  {
    "timestamp": "2026-11-09T13:10:27.538437+00:00",
    "request_id": "3a86fed8c54b4518b8686deb717f3230",
    "event": "output_delivered",
    "status": "found",
    "count": 8
  }
]
```

## Выводы и ограничения

В таблицах извлечённые параметры и найденные товары сопоставлены с ожидаемыми
результатами, рассчитанными по исходным данным. Ответы модели без доступа к базе
сравниваются с фактическим содержимым каталога: такие ответы сами по себе
не подтверждают наличие товаров.

Передача текущих параметров позволяет явно задать действующие ограничения.
Полная история сохраняет формулировки пользователя, но увеличивает объём контекста
с каждым ходом. Сравнение числа входных токенов и корректности ответов показывает,
как оба способа работают на выбранных диалогах.

Ошибки синтаксиса JSON и нарушения схемы различаются по типу исключения и его
причине. Валидация не допускает ошибочные параметры до поиска. Повторная генерация
позволяет продолжить обработку, если модель исправляет ответ; после исчерпания
попыток приложение возвращает статус `parse_error`. Команда `make verify` проверяет
полноту результатов, соответствие ожиданиям и порядок событий в журнале.

## Ответы на контрольные вопросы

1. В этой работе порядок действий задаёт программа. В автономном агенте модель
   может выбирать инструменты и решать, как продолжить работу с их результатами.
2. LCEL соединяет компоненты цепочки оператором `|`. `StrOutputParser` преобразует
   сообщение модели в строку; при прямом вызове модель возвращает `AIMessage`.
3. Системное сообщение `system` задаёт роль модели, правила обработки и формат
   ответа. Сообщение `human` содержит запрос пользователя.
4. Структурированный ответ соответствует заданной схеме и поддаётся программной
   проверке. Свободный текст не гарантирует нужную структуру и типы данных.
5. `BaseModel` описывает модель данных. `Field` задаёт описания, ограничения
   и значения по умолчанию, а `Literal` ограничивает набор допустимых значений.
6. Без доступа к SQLite модель не знает, какие товары записаны в этой базе.
   Знание распространённых моделей устройств не подтверждает их наличие в каталоге.
7. При вызове инструмента схема передаётся через API, а серверный парсер выделяет
   вызов и его аргументы. Приложение проверяет их и вызывает Python-функцию.
   При использовании форматных инструкций приложение само разбирает текст ответа.
8. Валидация ограничивает поиск допустимыми параметрами. SQL, сгенерированный
   моделью без ограничений, может изменить или раскрыть данные. Здесь SQL задаётся
   программой, а значения пользователя передаются через параметры.
9. Повторяются ошибки разбора JSON и проверки схемы. Обычный повтор снова
   использует тот же промпт. Исправление ответа с учётом ошибки дополнительно
   передаёт модели предыдущий ответ и описание обнаруженной проблемы.
10. Состояние хранится в программе. В следующий промпт передаётся либо текущий
    объект параметров, либо история сообщений. Веса модели между вызовами
    не обновляются.

## Источники

- [Условие лабораторной работы](../SPECIFICATION.md).
- [LangChain: ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai).
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
- [Стандартный модуль sqlite3](https://docs.python.org/3.12/library/sqlite3.html).
