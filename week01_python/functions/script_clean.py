def script_clean(text_input):

    import string

    text_cleaned = ''

    #remove punctuation
    for i in range(len(text_input)):
        if text_input[i] not in string.punctuation:
            text_cleaned += text_input[i]
        
    #Remove all white space and rejoin with individual spaces
    text_cleaned = text_cleaned.split()
    text_cleaned = ' '.join(text_cleaned)

    #Make lower case
    text_cleaned = text_cleaned.lower()

    return text_cleaned
