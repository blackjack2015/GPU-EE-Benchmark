# -*- coding: utf-8 -*-
from __future__ import print_function
from re import sub


def encode(text):
    return sub(r'(.)\1*', lambda m: str(len(m.group(0))) + m.group(1), text)

def decode(text):
    return sub(r'(\d+)(\D)', lambda m: m.group(2) * int(m.group(1)), text)

if __name__ == '__main__':
    value = encode("aaaaahhhhhhmmmmmmmuiiiiiiiaaaaaa")
    print("Encoded value is {}".format(value))
    print(decode(value))
