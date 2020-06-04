from unidecode import unidecode

def norm_string(input):
    text = input.strip().lower()
    text = unidecode(text)

    return text