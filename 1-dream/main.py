#!/usr/bin/env python3

import load, dream, store

import os
import sys

def dream_with_image_at_path(path_image):
    image = load.image(path_image)
    image_augmented = dream.with_image(image)
    store.image(image_augmented)

if '__main__' == __name__:
    args = sys.argv
    assert len(args) == 2, '{} takes a single path to image file as argument'.format(args[0])
    dream_with_image_at_path(args[1])