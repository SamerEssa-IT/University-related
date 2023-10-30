import yaml
from pathlib import Path
import os
import glob
import re



def remove_duplicates(Test_string):
    ## this function is useful to clean the translation errors that occured with back translation 
    Pattern = r"\b(\w+)(?:\W\1\b)+"
    Test_string = re.sub(r'\.\.+', ' ', Test_string)
    Test_string = re.sub(r'\-\s+', ' ', Test_string)
    return re.sub(Pattern, r"\1", Test_string, flags=re.IGNORECASE)

def funcutf8(text):
    return text.encode("latin-1").decode("UTF-8")

def get_txt_from_dir(path_dir :str ):
    os.chdir(path_dir)
    list_of_files = []
    for file in glob.glob("*.txt"):
        list_of_files.append(path_dir + "/" + file)
    return list_of_files

def main():
    # german sample texts:
    text = 'Klinische Angaben: Jejunocolische Anastomose; Colitis Ulcerosa (cw) Eingesandt wurde: 1 (Darm): ' \
           'Fragmentiertes, zusammengelegt yy x yy x yy cm großes Weichgewebe. Exemplarische Schnitte. Technik: ' \
           'HE-Färbung Histologie: 1: Darmwandanteile mit einem ausgedehnten, ulcerösen, oberflächlich deutlich eitrig, ' \
           'chronisch-granulierenden Defekt in der Schleimhaut. Dieser reicht bis an die Tunica muscularis heran. Daneben ' \
           'entzündlich-alterierte Dickdarmschleimhaut und architektonisch auffällige, kolonisierte Dünndarmschleimhaut. ' \
           'Beurteilung: 1: Resektat von einer jejunocolischen Anastomose mit einer bis an die Tunica muscularis ' \
           'reichenden, flächenhaften Ulzeration mit chronisch granulierender sowie eitriger Entzündung. Kein Anhalt für ' \
           'Malignität. Mit freundlichen Grüßen Prof. Dr. med. xx Dr. med. xx, M.Sc. (Tel.: xxxxx) '

    sections = get_report_sections(text)

    section_names = ['clinical infos:', 'sample infos:', 'desciption:\t', 'diagnosis:\t', 'end:\t\t\t']
    for i, section in enumerate(sections):
        print(f'{section_names[i]}\t\t{section}')


if __name__ == '__main__':
    main()
