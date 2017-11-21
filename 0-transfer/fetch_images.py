import urllib
from urllib.request import urlopen
import cv2
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool 
from functools import partial
import itertools
import pandas as pd
import os
from scipy import ndimage
import shutil
import uuid

np.random.seed(6)

pic_num = 1

def store_raw_images(paths, links):
    global pic_num
    for link, path in zip(links, paths):
        if not os.path.exists(path):
            os.makedirs(path)
        image_urls = str(urlopen(link).read())
        pool = ThreadPool(32)
        prod_x = partial(loadImage_star) # prod_x has only one argument x (y is fixed to 10) 
        l = zip(itertools.repeat(path),[s.strip() for s in image_urls.splitlines()],itertools.count(pic_num))
        pool.map(prod_x, l) 
        pool.close() 
        pool.join()
        
def loadImage_star(args):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return loadImage(*args)
    
def loadImage(path,link, counter):
    global pic_num
    if pic_num < counter:
        pic_num = counter+1;
    try:                
        urllib.urlretrieve(link, path+"/"+str(counter)+".jpg")
        img = cv2.imread(path+"/"+str(counter)+".jpg")             
        if img is not None:
            cv2.imwrite(path+"/"+str(counter)+".jpg",img)
            print(counter)

    except Exception as e:
        print(str(e))  
    
def removeInvalid(dirPaths):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            for invalid in os.listdir('invalid'):
                try:
                    current_image_path = str(dirPath)+'/'+str(img)
                    invalid = cv2.imread('invalid/'+str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).any()):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))
                    current_image_path = str(dirPath)+'/'+str(img)
                    os.remove(current_image_path)
  
  
def generate_dataset(paths, convertion):
    number_of_images = 0
    percentage = 0.3
    
    shutil.rmtree('data/training', ignore_errors=True) # Remove old dir
    shutil.rmtree('data/validation', ignore_errors=True) # Remove old dir
    
    for c in set(convertion):
        os.makedirs('data/training/'+ c) # Create new one
        os.makedirs('data/validation/'+ c) # Create new one
    
    k = 0
    for j in range(0, len(paths)):
        images = [f for f in os.listdir(paths[j]) if f.endswith('.jpg')]
        np.random.shuffle(images)
        split_index = int(len(images) * (1.0-percentage))
        train = images[0:split_index]
        validation = images[split_index:] 
        
        
        
        image_dst = 'data/training/' + convertion[j]
        for i in range(0, len(train)):
            image_src = paths[j] + '/' + train[i]
            shutil.copy2(image_src, image_dst + '/' + str(k+1) + ".jpg")
            k += 1
            
            
        image_dst = 'data/validation/' + convertion[j]
        for i in range(0, len(validation)):
            image_src = paths[j] + '/' + validation[i]
            shutil.copy2(image_src, image_dst + '/' + str(k+1) + ".jpg")
            k += 1
            

def main():
    # links = [ 
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09991867', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105', \
    #         'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537' ]
    # 
    # paths = ['pet', 'furniture', 'people', 'food', 'frankfurter', 'chili-dog', 'hotdog']
    
    links = [ 
            'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09991867', \
            'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
            'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
            'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537' ]
    
    hotdog = 'hottog'
    nothotdog = 'nothotdog'
    paths = ['pet', 'people', 'food','hotdog']
    convertion = [nothotdog, nothotdog, nothotdog, hotdog]
    
    # store_raw_images(paths, links)
    # removeInvalid(paths)
    generate_dataset(paths, convertion)


if __name__ == "__main__":

    main()
