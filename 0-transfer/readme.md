# Transfer learning

sudo pip3 install keras tensorflow matplotlib h5py pillow


Denne oppgaven går ut på å lage den legendariske appen "hotdog or not-hotdog".
Dataen vi skal bruke er hentet ned ImageNet(http://www.image-net.org/) og består av pølser, dyr, mennesker og mat.
For at scriptet vårt skal kunne lese dataen med Keras ImageDataGenerator må trening og valideringsdata være på et bestemt format.
--data
----training/
------Hotdog
------Not-hotdog
----validation
------Hotdog
------Not-hotdog


Scripts:
fine_tuning.py
- Loads image-generators and trains model. After training is finished the model is saved.

predict.py
- Loads model and predicts image from params.

Task:
Create a Keras model that classifies images in fine_tuning.py
