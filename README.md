# Running Efficiency Lab — Streamlit + screenshot extraction

Web app per stimare nel tempo la velocità equivalente a condizioni statisticamente comparabili. Non è una misura di running economy da laboratorio.

## Funzioni principali

- Upload di screenshot Strava, Garmin, Apple Fitness e app simili.
- Estrazione vision dei campi visibili: data, durata, distanza, passo medio, FC media, dislivello, potenza, cadenza e RPE.
- Nessuna invenzione di campi mancanti: ciò che non è leggibile resta vuoto.
- Revisione manuale obbligatoria del dataset prima del fit.
- Import CSV e inserimento manuale alternativi.
- Conversione passo → velocità e data → giorni dalla prima osservazione.
- Regressione robusta, verifica della non-linearità FC e stress test di parsimonia per distanza/dislivello.
- `v_eq` alle condizioni mediane del dataset.
- Trend flessibile vs isotono, residui, MAD, anomalie/downweighting.
- Forecast saturante confrontato con alternative semplici, validazione temporale e bootstrap con refit completo.
- Simulatori: passo sostenibile in una data/FC/distanza e inversione per stimare una data-obiettivo.

## Avvio locale

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Per l'estrazione screenshot imposta `OPENAI_API_KEY` come variabile d'ambiente oppure crea localmente `.streamlit/secrets.toml` partendo da `.streamlit/secrets.toml.example`.

## Pubblicazione online

Vedi `DEPLOY_ONLINE.md`. La cartella è già predisposta per Streamlit Community Cloud con `requirements.txt`, `.streamlit/config.toml`, `.gitignore` e un esempio di secrets.

## Regola “non inventare dati”

L'estrattore vision non calcola né stima campi mancanti. La tabella di revisione contiene quindi solo valori estratti/importati o inseriti manualmente. Dopo la conferma, se il passo manca ma sono presenti durata e distanza, il motore può derivarlo deterministicamente come `durata / distanza`; la app segnala quante righe usano questa derivazione e non la presenta come dato estratto.

## Definizione di v_eq

`v_eq` è il punto di efficienza statistico di ogni corsa: la velocità osservata corretta per riportarla alle condizioni di riferimento mediane di FC e, quando il modello le mantiene, distanza e dislivello. Non è running economy misurata in laboratorio.

## Parsimonia

Potenza, cadenza e RPE sono conservate ma non entrano automaticamente nel modello. Distanza/dislivello vengono rimossi se un modello più semplice mantiene una validazione temporale simile. La non-linearità della FC viene considerata solo con un campione sufficiente e se il miglioramento giustifica la complessità.

## Forecast

Il candidato principale è:

`g(t) = g∞ - (g∞ - g0) exp(-k t)`

ed è confrontato con trend lineare e `sqrt(time)`. `g∞` è il plateau implicato dalla traiettoria osservata e dal modello, non un limite genetico. `τ=1/k` è la scala temporale del processo saturante.

Il bootstrap ricostruisce pseudo-dataset sulla scala della velocità e rifitta l'intera pipeline a ogni replica. Fuori dal periodo osservato viene aggiunta una componente di incertezza crescente con l'orizzonte per evitare bande artificialmente piatte.

## Limiti attuali

- L'estrazione screenshot richiede una API key configurata dal proprietario della app.
- Non c'è ancora autenticazione personale o database persistente.
- Il modello resta osservazionale: vento, superficie, temperatura, fatica, percorso e qualità sensori possono restare confondenti.
- Con campioni piccoli, plateau e non-linearità possono essere scarsamente identificati.
