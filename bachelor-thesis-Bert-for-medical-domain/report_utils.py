import yaml
from pathlib import Path
from text_utils import regexp, google_translation, funcutf8
import os

path2keywords = f'{os.path.dirname(__file__)}/patho_section_keywords.yaml'
start_keywords = yaml.safe_load(Path(path2keywords).read_text())


def get_report_sections(t_text: str, combine_all_conclusions=True) -> tuple[str, str, str, str, str]:
    """
    Divides the given text into text sections that typically occur in pathological reports.
    :param combine_all_conclusions: If there are multiple conclusions in t_text,
    concatenate them together? Otherwise, just use the last conclusion
    (not recommended, since the last conclusion might be a Nachbericht).
    :param t_text: The pathological text.
    :return: Returns these sections as strings:
    [clinic_infos, sample_infos, description, conclusion, greetings].
    A section is None if it could not be found
    """
    
    try:
        t_text = func_utf8(t_text)
    except Exception as E:
        pass
    
    def find_codon(text, word_list):
        '''
        returns the word of word_list, which could be found first in text.
        If no word could be found, returns None.
        '''

        for i_word in word_list:
            i_word = i_word
            if text.find(i_word) > -1:
                codon = i_word
                return codon

        return None

    # thinking that one pathologist sticks to his/her wording:
    start_codon_clinical_infos = find_codon(t_text, start_keywords['clinical_infos'])

    start_codon_sample_infos = find_codon(t_text, start_keywords['sample_infos'])

    start_codon_description = find_codon(t_text,
                                         start_keywords['description'])
    start_codon_description_2nd = find_codon(t_text,
                                             start_keywords['follow-up-report'])

    start_codon_conclusion = find_codon(t_text,
                                        start_keywords['conclusion'])
    # Vorläufige Beurteilung gemäß der Gefrierschnittführung: .... Beurteilung am Paraffinmaterial:

    start_codon_comment = find_codon(t_text,
                                     start_keywords['comment'])

    start_codon_greetings = find_codon(t_text,
                                       start_keywords['regards'])

    # like on DNA, the next start codon is a stop codon
    stop_codon_list = [start_codon_conclusion, start_codon_description,
                       start_codon_comment, start_codon_greetings, start_codon_description_2nd,
                       start_codon_clinical_infos, start_codon_sample_infos]

    def get_codon_idx(text, start_codon, stop_codon_list):

        if not start_codon:
            return [], []

        _, idx_start = regexp(start_codon, text)

        idx_stop = []
        for i_idx_start in idx_start:

            idx_stop_list = []
            for i_stop_codon in stop_codon_list:
                if i_stop_codon == None:
                    continue
                if not i_stop_codon == start_codon:
                    idx_stop_list.append(text[i_idx_start:].find(i_stop_codon))

            idx_stop_list = [item for item in idx_stop_list if item >= 0]
            if idx_stop_list:
                idx_stop.append(min(idx_stop_list) + i_idx_start)
            else:
                return [], []

        return idx_start, idx_stop

    # %% find the indices for the text-frames
    start_clinical_infos, stop_clinical_infos = get_codon_idx(t_text,
                                                              start_codon_clinical_infos,
                                                              stop_codon_list)

    start_sample_info, stop_sample_info = get_codon_idx(t_text,
                                                        start_codon_sample_infos,
                                                        stop_codon_list)

    start_description, stop_description = get_codon_idx(t_text,
                                                        start_codon_description,
                                                        stop_codon_list)

    start_2nd, stop_2nd = get_codon_idx(t_text,
                                        start_codon_description_2nd,
                                        stop_codon_list)

    start_conclusion, stop_clonclusion = get_codon_idx(t_text,
                                                       start_codon_conclusion,
                                                       stop_codon_list)

    # %% get the text parts
    def get_text_frame(idx_start_list, idx_stop_list, text):
        t_frame = []
        for i in range(0, len(idx_start_list)):
            t_frame.append(text[idx_start_list[i]:idx_stop_list[i]])

        return t_frame

    txt_clinic_infos = get_text_frame(start_clinical_infos, stop_clinical_infos, t_text)

    txt_sample_infos = get_text_frame(start_sample_info, stop_sample_info, t_text)

    txt_description = get_text_frame(start_description, stop_description, t_text)

    txt_description_2nd = get_text_frame(start_2nd, stop_2nd, t_text)

    txt_conclusion = get_text_frame(start_conclusion, stop_clonclusion, t_text)

    # get greetings-section:
    txt_greetings = None
    if start_codon_greetings:
        start_greedingsindex = t_text.find(start_codon_greetings)
        if start_greedingsindex != -1:
            txt_greetings = t_text[start_greedingsindex:]

    # construct description
    if txt_description_2nd:
        txt_description_2nd_str = start_codon_description_2nd + ' ' + txt_description_2nd[0]
        if txt_description:
            txt_description = str(txt_description[0]) + str(txt_description_2nd_str)
        else:
            txt_description = str(txt_description_2nd_str)
    else:
        if txt_description:
            txt_description = str(txt_description[0])
        else:
            txt_description = None

    # construct diagnose/conclusion:
    if txt_conclusion:
        if combine_all_conclusions:
            txt_conclusion_appended = ''
            for conc_text in txt_conclusion:
                txt_conclusion_appended += conc_text
            txt_conclusion = txt_conclusion_appended
        else: # use last conclusion:
            txt_conclusion = txt_conclusion[-1]
    else:
        txt_description = None

    return txt_clinic_infos[0] if txt_clinic_infos else None, \
           txt_sample_infos[0] if txt_sample_infos else None, \
           txt_description, \
           txt_conclusion, \
           txt_greetings


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
