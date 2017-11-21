# Transfer learning

## Requirements: 
pip install keras==2.0 tensorflow==1.0.1 h5py==2.7.0 matplotlib==1.5.3 scipy==0.19.0 pillow

Plotting fungerer ikke ved bruk av virtual env - så denne koden må kopieres ut. 

Scriptene skal fungere med python 2.7 og 3.x, men har opplevd noen problemer med 3.x.  


## Oppgave:
Create a Keras model that classifies images in fine_tuning.py
This should be implemented in create_model (fine_tuning.py)

Denne oppgaven går ut på å lage den legendariske appen "hotdog or not-hotdog" (https://www.youtube.com/watch?v=ACmydtFDTGs).
Dataen vi skal bruke er hentet ned ImageNet(http://www.image-net.org/) og består av pølser, dyr, mennesker og mat.
For at scriptet vårt skal kunne lese dataen må trening og valideringsdata være på et bestemt format. I denne oppgaven skal vi ta i bruk et eksisterende nettverk (https://keras.io/applications/#vgg16) og fin-tune dette til å klassifisere pølser/ikke pølser. Dette kan gjøres ved å fjerne vgg16-modellens X-siste lag og legge til våre egne. Lagene vi legger til kan vi fortsette å trene og dermed fin-tune modellen til vårt domene. 

--data<br />
----training<br />
------Hotdog<br />
------Not-hotdog<br />
----validation<br />
------Hotdog<br />
------Not-hotdog<br />


## Scripts:
fine_tuning.py
- Lastet bilder og trener modellen. Etter modellen er trent vil modellen bli lagret. 

predict.py
- Laster modellen og predikerer bildet gitt i paramenter. 
Eksempel: python predict.py bilder/pølse.jpg
