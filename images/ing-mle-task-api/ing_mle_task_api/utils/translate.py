def translate_category(ctg: str) -> str:
    if ctg == 'b':
        return 'business'
    elif ctg == 't':
        return 'science and technology'
    elif ctg == 'e':
        return 'entertainment'
    elif ctg == 'm':
        return 'health'
    else:
        return 'unknown'