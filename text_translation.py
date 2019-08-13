from googletrans import Translator

# Translates from hebrew into english:
def translate_list(source_list, translated_list):
    translator = Translator()

    for i in range(len(source_list)):
        translation = translator.translate(source_list[i])
        translated_list.append(translation.text)