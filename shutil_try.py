from shutil import copyfileobj as cfj
import sys
filename = r'D:\Тут_был_хлебушек\datadir\processed_text.test.txt'
with open(filename, encoding='utf-8') as file:
    text = file.read()

filename_write = r'D:\Тут_был_хлебушек\datadir\processed_text2.test.txt'
with open(filename_write, mode='w', encoding='utf-8') as f:
    f.write(text)
    f.close()