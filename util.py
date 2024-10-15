import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def license_complies_format(text):
    """
    Check if format complies
    """
    if len(text) != 7:
        return False
    
    special_chars='!@#$%^&*"()?<>/'
    
    if any(s in special_chars for s in text):
        return False
    
    return True


def format_license(text):
    """
    Alphanumeric character conversion based on position
    """
    # Alphanumeric character conversion list
    dict_char_to_int = {'O': '0',
                        'I': '1',
                        'J': '3',
                        'A': '4',
                        'G': '6',
                        'S': '5',
                        'B': '8',
                        'E': '6',
                        'T': '1'}

    dict_int_to_char = {'0': 'O',
                        '1': 'I',
                        '2': 'Z',
                        '3': 'J',
                        '4': 'A',
                        '5': 'S',
                        '6': 'G',
                        '7': 'T',
                        '8': 'B',
                        '9': 'B'}


    license_plate_ = ''
    if len(text) >= 4:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}
        for j in range(len(text)):
            if j in mapping and text[j] in mapping[j]:
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j] 
        return license_plate_
    return None


def read_license_plate_easyocr(license_plate_crop):
    """
    Easyocr model
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
    
    return None, None