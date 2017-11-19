# Drømming
Denne oppgaven baserer seg på kode tilsvarende [Google's forsøk](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) en stund tilbake. Målet er å gjøre endringer i et bilde for å tilfredsstille et sett nevroner. Med andre ord skal vi fremprovosere mønstre i bildet nevroner i nettverket ser etter. Intensjonen med oppgaven er å gjøre noe litt annerledes, men dette kan også brukes for debugging av nettverket for å se hva det egentlig har plukket opp.

Koden kan kjøres som den er: `./main.py <sti_til_bildefil>`. Eksempelbilder ligger i mappen `resource`. Du står fritt til å bruke egne bilder, men vit at bildene blir skalert ned til 224x224 piksler for å passe til nettverket. Omtrent kvadratiske bilder med få små detaljer fungerer derfor bedre. Output fra koden havner i `out.png`.

Prøv først å kjøre koden, så vi vet det funker: `./main.py resource/trump.jpg`
Se resultatet i `out.png`. I resultatbildet har vi optimalisert for alle nevroner i laget som heter `block1_conv1`. Vi kan bytte til et senere lag ved å endre `layer_to_optimize = 'block1_conv1'` til f.eks `layer_to_optimize = 'block2_conv1'`. Senere lag er som vi vet mer abstrakte, og vil vise mer komplekse mønstre.

Vi kan også velge å optimalisere for et spesifikt type mønster ved å sette `filters_to_optimize = <feature>`, hvor `<feature>` er indeksen til et mønster nettverket ser etter i laget vi har spesifisert med `layer_to_optimize`. Eksempel: `filters_to_optimize = 5`. Ved å sette `filters_to_optimize = None` vil alle mønstre innenfor laget bli optimert.

I funksjonen `augment_image` kan du justere `scale` og `niterations` for å justere henholdsvis hvor fort endringer blir gjort i bildet og hvor mange ganger det skjer.

De ulike lagene kan du finne [her](https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py), men kort fortalt: `block{1,2,3,4,5}_conv{1,2,3}`.

Have fun!