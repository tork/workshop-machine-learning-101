# Maskinlæring 101
Denne workshopen tar for seg maskinlæring, eksemplifisert gjennom nevrale nettverk og [Tensorflow](https://www.tensorflow.org).

Workshopen består av flere moduler, organisert i nummererte kataloger.

## Forberedelser
Miljøet må gjøres klart, og data lastes ned på forhånd.

### Miljø
Workshopen tar utgangspunkt i Python og Tensorflow. Som et forsøk på å minimere feil hos deltakerne, har jeg lagt inn en kommando som kjører opp et [Docker](https://www.docker.com)-image. Installer Docker. Dersom du kjører Windows, må du godkjenne partisjonen prosjektet ligger på for deling:
Høyreklikk på Docker-ikonet i taskbaren, gå til settings. Under "Shared Drives", huk av partisjonen som inneholder koden.
Åpne en terminal og kjør `env.sh` eller `env.ps1` for henholdsvis Linux/macOS eller Windows.

Sjekk at containeren starter uten problemer, og at du kommer inn i et shell. Verifiser at prosjektmappen ligger tilgjengelig på `/workshop-machine-learning-101`. Det er viktig at dette steget blir utført i forkant av workshopen, slik at du har imaget klart lokalt.

### Data
Datasettene som blir brukt lastes ned ved å kjøre `data.sh`. Scriptet er kjørbart fra utviklermiljøet i Docker-containeren fra steget over, men bør også fungere i macOS. Sjekk at data lastes ned uten feil, og at scriptet avslutter ved å skrive `done`.
