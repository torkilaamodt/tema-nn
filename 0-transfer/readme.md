# Transfer learning

## Requirements: 
sudo pip install keras tensorflow matplotlib h5py pillow

Skal fungere med python 2.7 og 3.x, men har opplevd at 2.7 er raskere. 



Denne oppgaven går ut på å lage den legendariske appen "hotdog or not-hotdog" (https://www.youtube.com/watch?v=ACmydtFDTGs).
Dataen vi skal bruke er hentet ned ImageNet(http://www.image-net.org/) og består av pølser, dyr, mennesker og mat.
For at scriptet vårt skal kunne lese dataen med Keras ImageDataGenerator må trening og valideringsdata være på et bestemt format.

--data<br />
----training<br />
------Hotdog<br />
------Not-hotdog<br />
----validation<br />
------Hotdog<br />
------Not-hotdog<br />


## Scripts:
fine_tuning.py
- Loads image-generators and trains model. After training is finished the model is saved.

predict.py
- Loads model and predicts image from params.

## Task:
Create a Keras model that classifies images in fine_tuning.py
This should be implemented in create_model (fine_tuning.py)
