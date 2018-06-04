from glob import glob
import os

imgs_paths = glob('C:\\Users\\Lee Twito\\Desktop\\NORMAL 1-50\\*.bmp')

for path in imgs_paths:
    print(path)
    os.rename(path, path[:-3] + 'png')
