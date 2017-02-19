# Maskinlæring 101
Denne workshopen tar for seg maskinlæring, eksemplifisert gjennom nevrale nettverk og [Tensorflow](https://www.tensorflow.org).

Workshopen består av flere moduler, organisert i nummererte kataloger.

## Forberedelser
Miljøet må gjøres klart, og data lastes ned på forhånd.

### Klon repo
Vi kommer til å kjøre koden i Linux. Siden Windows bruker et ekstra tegn for å indikere slutten på en linje, må Windows-brukere kjøre følgende kommando før repo clones:
`git config --global core.autocrlf false`
Git har mulighet til å legge på det ekstra tegnet automatisk når man cloner et repo. Dette er ikke ønskelig her, så vi deaktiverer det ved å si `false` som angitt over.

Nå kan du clone koden: `git clone https://github.com/tork/workshop-machine-learning-101.git`

### Docker
Workshopen tar utgangspunkt i Python og Tensorflow. For å ha bedre kontroll på utviklermiljøet, er det lagt opp til bruk av Docker lokalt. Om mulig, installer en native Docker-versjon. Eldre versjoner av Windows og macOS må kjøre Docker virtuelt, med Docker Toolbox. Noen nye versjoner av Windows mangler også Hyper-V, og vil ikke kunne kjøre Docker native. Har du ikke Windows 10 Professional/Enterprise 64-bit eller bedre, må Docker Toolbox benyttes.

NB: Om Docker Toolbox skal benyttes, trenger man en driver for virtualisering. I utgangspunktet er dette Virtualbox, og installeres sammen med Docker Toolbox. Noen Windows vil ha Hyper-V installert, og Docker Toolbox støtter å bruke Hyper-V som driver. Selv opplevde jeg noen problemer med den kombinasjonen. Hvis du møter problemer med Hyper-V og Docker Toolbox, prøv å deaktiver Hyper-V og følg resten av readme-en som vanlig.

Installer Docker eller Docker Toolbox. Dersom du kjører Windows, må du godkjenne partisjonen prosjektet ligger på for deling:
Høyreklikk på Docker-ikonet i taskbaren, gå til settings. Under "Shared Drives", huk av partisjonen som inneholder koden.

Åpne prosjektmappen og kjør et av env-scriptene, avhengig av konfigurasjon:
OS|Docker|Script
Linux/macOS|Native|env-native.sh
Linux/macOS|Toolbox|env-toolbox.sh
Windows|Native|env-native.ps1
Windows|Toolbox|env-toolbox.ps1

`env-toolbox.ps1` må kjøres som administrator (start Powershell som administrator og kjør scriptet derfra). Årsaken er at man trenger admin for å hente info om Hyper-V.

Sjekk at containeren starter uten problemer, og at du kommer inn i et shell. Verifiser at prosjektmappen ligger tilgjengelig på `/workshop-machine-learning-101`. Det er viktig at dette steget blir utført i forkant av workshopen, slik at du har imaget klart lokalt.

### Data
Datasettene som blir brukt lastes ned ved å kjøre `data.sh`. Scriptet er kjørbart fra utviklermiljøet i Docker-containeren fra steget over, men skal også fungere i macOS. Sjekk at data lastes ned uten feil, og at scriptet avslutter ved å skrive `done`.
