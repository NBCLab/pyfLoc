import time


CODE = {
    ' ': '_',
    "'": '.----.',
    '(': '-.--.-',
    ')': '-.--.-',
    ',': '--..--',
    '-': '-....-',
    '.': '.-.-.-',
    '/': '-..-.',
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    ':': '---...',
    ';': '-.-.-.',
    '?': '..--..',
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '_': '..--.-'
}


def convert_to_morse(message):
    """
    From https://ispycode.com/Blog/python/2016-07/Convert-Text-To-Morse-Code
    """
    message = message.upper()
    morse_message = ''
    for character in message:
        morse_message += CODE[character] + ' '
    return morse_message


def biopac_signature(ser, message):
    """
    Write a binary message using the serial port in morse code.
    """
    ser.write('RR')
    message = convert_to_morse(message)
    dot_duration = 0.1
    dash_duration = 0.2
    space_duration = 0.3
    sep_duration = 0.05
    lead_duration = 0.2

    time.sleep(lead_duration)
    for char in message:
        if char == '-':
            ser.write('FF')
            time.sleep(dash_duration)
        elif char == '.':
            ser.write('FF')
            time.sleep(dot_duration)
        elif char == ' ':
            time.sleep(space_duration)
        else:
            raise Exception('Unknown character "{}"'.format(char))
        ser.write('RR')
        time.sleep(sep_duration)
