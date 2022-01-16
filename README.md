# Восстановление пунктуации для русского языка

В этом репозитории представлена модель, расставляющая знаки препинания (точки, запятые и вопросительные знаки) для текстов, генерируемых ASR-системами. Она основана на tensorflow-версии BRNN with Attention Mechanism, предложенной Ottokar Tilk (https://github.com/ottokart/punctuator2), но обучена на русском корпусе текстов (оригинальная архитектура была обучена на английском и эстонском, ее последующие имплементации - на исландском). Модель превосходит имеющиеся для русского языка решения, основанные на BERT (имеются в виду https://github.com/sviperm/neuro-comma и https://github.com/snakers4/silero-models; мы оцениваем их по проценту точности запятых, потому что эти модели не сегментируют текст на предложения: точность расстановки точек несопоставима). 

# Применение обученной модели

Клонируйте или скачайте репозиторий. Переместитесь в папку проекта. Создайте виртуальную среду с Python 3.8.0 (мы рекомендуем через "python -m virtualenv -p PATH\TO\python.exe ENVIRONMENT_NAME"). Затем активируйте среду и установите библиотеки:

```
pip install -r requirements.txt
```
<a href='https://drive.google.com/file/d/1ArzZuKFVyjriVYxKSHsOU5BKuSLVN1Ko/view?usp=sharing'> Скачайте</a> модель из Google Drive и поместите в папку проекта.
Запустите файл playing_with_model.py:
```
python playing_with_model.py
```

В строку ввода (после слов "Введите текст без знаков пунктуации") скопируйте или введите текст, лишенный знаков препинания. Он появится в строке вывода с расставленной пунктуацией.
<br>Важно! Предметная область модели не универсальна: модель обучена на новостных текстах с Lenta.ru, поэтому не ожидайте отличных результатов при пунктуации, скажем, стихотворений.

Пример текстов, откорректированных моделью, до и после:

| До восстановления                                         | После восстановления |
|-----------------------------------------------------------|----------------------|
| израильский министр по делам диаспоры нахман шай сообщил что пристально следит за развитием событий в американском городе колливилль штат техас где неизвестный захватил заложников в синагоге по сообщению американского телеканала abc news захвативший заложников в синагоге в колливилле удерживает четверых человек и требует освобождения женщины осужденной за покушение на убийство американского солдата пока известно только об одном подозреваемом указывает телеканал по данным сми захвативший заложников заявляет что является братом осужденной за терроризм аафии сиддики и требует освобождения сестры она обвинялась в связях с "аль-каидой"* и была осуждена на 86 лет за нападение и покушение на убийство американского солдата а сейчас содержится в заключении на базе bbc недалеко от форт-уэрт в техасе | Израильский министр по делам диаспоры нахман шай сообщил, что пристально следит за развитием событий в американском городе колливилль, штат техас, где неизвестный захватил заложников в синагоге. По сообщению американского телеканала abc news, захвативший заложников в синагоге в колливилле удерживает четверых человек и требует освобождения женщины, осужденной за покушение на убийство американского солдата, пока известно только об одном подозреваемом, указывает телеканал. По данным сми, захвативший заложников заявляет, что является братом осужденной за терроризм, аафии сиддики и требует освобождения сестры. Она обвинялась в связях с "аль-каидой"* и была осуждена на 86 лет за нападение и покушение на убийство американского солдата, а сейчас содержится в заключении на базе bbc недалеко от форт-уэрт в техасе. |
| в россии в 2019 году была запущена программа «дальневосточная ипотека» которая подразумевает выдачу кредитов под 2% на покупку жилья в Дальневосточном федеральном округе (дфо) программа направлена на улучшение жилищных условий в регионе и развитие местного строительного рынка продлится она до 2025 года рассказываем как получить льготный кредит под 2% годовых на дальнем востоке и на что его можно потратить | В россии в 2019 году была запущена программа «дальневосточная ипотека», которая подразумевает выдачу кредитов под 2% на покупку жилья в дальневосточном федеральном округе (дфо). Программа направлена на улучшение жилищных условий в регионе и развитие местного строительного рынка. Продлится она до 2025 года рассказываем, как получить льготный кредит под 2% годовых на дальнем востоке и на что его можно потратить |

# English

This repository contains a model that predicts punctuation marks (dots, commas and question marks) for output of ASR systems. It contains code of the tensorflow implementation of the BRNN with Attention Mechanism proposed by Ottokar Tilk (https://github.com/ottokart/punctuator2), but trained on a Russian corpus (the original model was trained on Emglish and Estonian datasets, its later reimplementations on Icelandic). The model outperforms similar solutions that are based on BERT (namely, https://github.com/sviperm/neuro-comma и https://github.com/snakers4/silero-models; we take into account comma restoration accuracy only, as these models do not segment text into sentences and therefore accuracy as per dots is incomparable).

# Running the pretrained model

Clone or download the repository, then move to the project directory. Create a virtual environment based on Python 3.8.0 (we recommend "python -m virtualenv -p PATH\TO\python.exe ENVIRONMENT_NAME"). Then activate the environment and install all dependencies:

```
pip install -r requirements.txt
```
<a href='https://drive.google.com/file/d/1ArzZuKFVyjriVYxKSHsOU5BKuSLVN1Ko/view?usp=sharing'>Download</a> the pretrained model from Google Drive and copy it to the project folder.
Run playing_with_model.py:

```
python playing_with_model.py
```

In the input line (after "Введите текст без знаков пунктуации") copy or type in a text lacking punctuation. It will be printed in the output line with punctuation restored.
<br>Note! The model is not universal in terms of domain: it was trained using news texts from Lenta.ru, therefore do not expect excellent results when punctuating, let's say, poems.
