# ML in Mechanics — переходный процесс сервопривода

Работа по теме «Линейная модель» на 167 наблюдениях UCI Servo.
Основной результат: [REPORT.md](REPORT.md). Протокол до обучения:
[protocol.md](protocol.md). Краткий аудит и устройство проекта:
[PROJECT_AUDIT.md](PROJECT_AUDIT.md).

## Запуск (Windows PowerShell, Python 3.14)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python run_analysis.py
.venv\Scripts\python predict.py --motor A --screw A --pgain 4 --vgain 2
```

Для первичного извлечения из архива вместо обычного запуска:

```powershell
.venv\Scripts\python run_analysis.py --archive "..\servo.zip"
```

Исходные data/raw уже включены: повторному запуску интернет не нужен после
установки библиотек. Полный nested CV занимает несколько минут (зависит от
процессора). Код однопоточный на уровне поиска для предсказуемого потребления
ресурсов. Точные версии зафиксированы в requirements.txt. Генерация отчёта
перезаписывает REPORT.md и results; исходные данные сохраняет без изменений.

```powershell
.venv\Scripts\python -m black --check *.py
.venv\Scripts\python -m flake8 . --exclude .venv
```

## Структура

```text
agents.md                 правила пользователя (не изменены)
protocol.md               гипотезы и решения до обучения
servo_model.py            базисы, Pipeline, сетки и размерностные проверки
run_analysis.py           весь эксперимент и создание отчёта/графиков
predict.py                собственный CLI-проект прогноза сервопривода
test_servo.py             pytest: пределы, размерности, отсутствие утечки
data/raw/                 оригинальные servo.data, servo.names, Index
results/                  метрики, OOF, коэффициенты, bootstrap, модель
results/figures/          диагностические графики
REPORT.md                 полный отчёт по заданию на русском
PROJECT_AUDIT.md          краткая памятка проекта и научного аудита
```

Вся вычислительная часть написана на Python. Итоговый прогноз линейный по
коэффициентам в исходной шкале y; log(y) — отдельное сравнение. motor/screw
категориальные. Препроцессинг внутри Pipeline; настройка не видит test.
Проверка переноса в pgain=3 независима от настройки обычной модели.

## Источник и единицы

[UCI Servo, Karl Ulrich, 1986](https://archive.ics.uci.edu/dataset/87/servo),
DOI: 10.24432/C5Q30F, лицензия CC BY 4.0. В исходных метаданных единицы не
заданы; вывод использует единицы набора данных, не секунды. Интерпретация
целевой величины как времени нарастания принята по условию задания с оговоркой
о неопределённости исторического описания.

Сохранённый joblib предназначен для доверенного локального использования с
указанными версиями библиотек. CLI отклоняет неизвестные категории и значения
вне документированного диапазона; внутри диапазона всё равно возможны
неизученные комбинации и большой локальный остаток.
