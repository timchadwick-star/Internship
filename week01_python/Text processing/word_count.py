from functions.script_clean import script_clean as clean
from functions.dict_count import dict_count as count
import csv

with open('./donald_speech.txt', 'r') as t:
    text = t.read()
    text = clean(text)
    word_count = count(text.split())

with open('./text_data.csv', 'w', newline='') as t:

    headers = ['Word','Count']

    writer = csv.DictWriter(t, fieldnames=headers)

    writer.writeheader()

    for word, count in word_count.items():
        writer.writerow({'Word': word, 'Count': count})
