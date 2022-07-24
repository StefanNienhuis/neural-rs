*[English](README.md) - Dutch*

---

# neural-rs

Een neuraal netwerk geschreven in Rust voor een schoolproject.

## Onderdelen

- `neural` - een algemene bibliotheek voor neurale netwerken
- `neural-cli` - een CLI-programma voor de neural bibliotheek die IDX-dataset-bestanden kan gebruiken
- `neural-emnsit` - een web interface voor de neural bibliotheek die neurale netwerken getraind met EMNIST kan gebruiken

### neural-emnist

`neural-emnist/src` is een Rust crate met WebAssembly bindings voor de neural bibliotheek.

Deze kan worden gecompileerd tot WebAssembly met `wasm-pack build --target web`.

`neural-emnist/www` is een web interface gebaseerd op Svelte die getallen en letters kan herkennen. Het standaard getallen netwerk is getraind met de EMNIST getallen dataset, en heeft een nauwkeurigheid van 99.06% over de test dataset. Het standaard letters netwerk is getraind met de EMNIST letters dataset, en heeft een nauwkeurigheid van 90.02% over de test dataset. De webinterface gebruikt de `pkg/` map uit de WebAssembly build als dependency.

Wanneer je op detecteren drukt, wordt het begrenzingsvak van de tekening berekend en wordt er een vierkant uitgehaald, deze wordt vervolgens geschaald naar een 28x28 pixels afbeelding via bilinear interpolation. Het direct tekenen op een canvas van 28x28 leidde tot een lagere nauwkeurig, doordat de tekening niet altijd gecentreerd stond.

## Voorbeelden

### EMNIST getallen

*Dit voorbeeld is te zien op [stefannienhuis.github.io/neural-rs/neural-emnist/](https://stefannienhuis.github.io/neural-rs/neural-emnist/).*

Maak een nieuw netwerk met 784 inputs en 10 outputs. De lagen tussenin en de kostenfunctie kan aangepast worden.
```shell
neural-cli create -l input:784 relu:300 relu:50 sigmoid:10 -c mean-squared-error ./network.nnet
```

Train het netwerk met de EMNIST getallen dataset (herhalingen: 30, leersnelheid: 0.3). De dataset is te verkrijgen van [de nist.gov website](https://www.nist.gov/itl/products-and-services/emnist-dataset).
```shell
neural-cli train -e 30 -r 0.3 --test-inputs ./test-images --test-labels ./test-labels ./network.nnet ./train-images ./train-labels
```

Open nu de `neural-emnist` webinterface en upload het nieuwe netwerk. Teken een aantal getallen om te zien hoe goed het werkt.

Als alles werkt, kun je de hyperparameters (lagen, herhalingen, leersnelheid etc...) aanpassen om te zien hoe dit de nauwkeurigheid beïnvloed.

De getallen dataset kan ook worden vervangen door de letters dataset, voor a-z herkenning. Hiervoor moet de output laag een grootte hebben van 27, omdat de index van de labels beginnen bij 1. Output 0 kan worden genegeerd. 

## Credits

Een groot gedeelte van het backpropagation algoritme is gebaseerd op de videoserie [*Neural networks*](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) gemaakt door 3Blue1Brown en het boek [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/index.html) geschreven door Michael Nielsen. 