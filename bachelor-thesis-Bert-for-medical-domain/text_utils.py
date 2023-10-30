import codecs
from googletrans import Translator  # use pip install googletrans==3.1.0a0, 3.0 version is broken

german_encoding = "latin-1"

def read_german_text(filename):

    textfile = codecs.open(filename, 'br', german_encoding)
    text= textfile.read()
    textfile.close()

    return text

def save_german_text(path, text):
    '''

    :param path: Where to save the .txt file
    :param text: Content of .txt file
    :return:
    '''
    with open(path, "w", encoding=german_encoding) as text_file:
        text_file.write(text)

def regexp(keyword: str, text: str):
    '''

    :param keyword: The keyword to search in text.
    :param text: Text to search.
    :return: returns index_start (as list) and index_stop (as list) of the keyword as 2 lists
    '''
    import re
    index_start, index_stop = [], []
    value = []
    for match in re.finditer(keyword, text):
        index_start.append(match.start())
        index_stop.append(match.end())
        value.append(match.group())

    return index_start, index_stop


def google_translation(text, src='de', to='en'):
    translator = Translator()
    return translator.translate(text, src=src, dest=to).text.replace('\u200b', '') # fix encoding bug

def funcutf8(text):
    return text.encode("latin-1").decode("UTF-8")