from imageio import imwrite

import os
from pathlib import Path

def image(image):
    path_store_dir = Path(os.path.abspath(__file__))/'..'
    path_out = path_store_dir/'out.png'
    imwrite(str(path_out), image)