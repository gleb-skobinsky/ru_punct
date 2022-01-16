# Восстановление пунктуации для русского языка

В этом репозитории представлена модель, расставляющая знаки препинания (точки, запятые и вопросительные знаки) для текстов, генерируемых ASR-системами. Она основана на tensorflow версии BRNN with Attention Mechanism, предложенной Ottokar Tilk (https://github.com/ottokart/punctuator2), но обучена на русском корпусе текстов (оригинальная архитектура была обучена на английском и эстонском, ее последующие имплементации - на исландском). Модель превосходит имеющиеся для русского языка решения, основанные на BERT (имеются в виду https://github.com/sviperm/neuro-comma и https://github.com/snakers4/silero-models; мы оцениваем их по проценту точности запятых, потому что эти модели не сегментируют текст на предложения: точность расстановки точек несопоставима). 

# Применение обученной модели

Клонируйте или скачайте репозиторий. Переместитесь в папку проекта. Создайте виртуальную среду с Python 3.8.0 (мы рекомендуем через python -m virtualenv -p PATH\TO\python.exe ENVIRONMENT_NAME). Затем активируйте среду.
Установите библиотеки:

```
pip install -r requirements.txt
```
<a href='https://drive.google.com/file/d/1ArzZuKFVyjriVYxKSHsOU5BKuSLVN1Ko/view?usp=sharing'> Скачайте</a> модель из Google Drive и поместите в папку проекта.
Запустите файл playing_with_model.py:
```
python playing_with_model.py
```

В строку ввода (после слов "Введите текст без знаков пунктуации") скопируйте или введите текст, лишенный знаков препинания. Он появится в строке вывода с расставленной пунктуацией.
Важно! Предметная область модели не универсальна: модель обучена на новостных текстах с Lenta.ru, поэтому не ожидайте отличных результатов при пунктуации, скажем, стихотворений.

Пример текстов, откорректированных моделью, до и после:

| До восстановления                                         | После восстановления |
|-----------------------------------------------------------|----------------------|
| израильский министр по делам диаспоры нахман шай сообщил что пристально следит за развитием событий в американском городе колливилль штат техас где неизвестный захватил заложников в синагоге по сообщению американского телеканала abc news захвативший заложников в синагоге в колливилле удерживает четверых человек и требует освобождения женщины осужденной за покушение на убийство американского солдата пока известно только об одном подозреваемом указывает телеканал по данным сми захвативший заложников заявляет что является братом осужденной за терроризм аафии сиддики и требует освобождения сестры она обвинялась в связях с "аль-каидой"* и была осуждена на 86 лет за нападение и покушение на убийство американского солдата а сейчас содержится в заключении на базе bbc недалеко от форт-уэрт в техасе | Израильский министр по делам диаспоры нахман шай сообщил, что пристально следит за развитием событий в американском городе колливилль, штат техас, где неизвестный захватил заложников в синагоге. По сообщению американского телеканала abc news, захвативший заложников в синагоге в колливилле удерживает четверых человек и требует освобождения женщины, осужденной за покушение на убийство американского солдата, пока известно только об одном подозреваемом, указывает телеканал. По данным сми, захвативший заложников заявляет, что является братом осужденной за терроризм, аафии сиддики и требует освобождения сестры. Она обвинялась в связях с "аль-каидой"* и была осуждена на 86 лет за нападение и покушение на убийство американского солдата, а сейчас содержится в заключении на базе bbc недалеко от форт-уэрт в техасе. |
| в россии в 2019 году была запущена программа «дальневосточная ипотека» которая подразумевает выдачу кредитов под 2% на покупку жилья в Дальневосточном федеральном округе (дфо) программа направлена на улучшение жилищных условий в регионе и развитие местного строительного рынка продлится она до 2025 года рассказываем как получить льготный кредит под 2% годовых на дальнем востоке и на что его можно потратить | В россии в 2019 году была запущена программа «дальневосточная ипотека», которая подразумевает выдачу кредитов под 2% на покупку жилья в дальневосточном федеральном округе (дфо). Программа направлена на улучшение жилищных условий в регионе и развитие местного строительного рынка. Продлится она до 2025 года рассказываем, как получить льготный кредит под 2% годовых на дальнем востоке и на что его можно потратить |

