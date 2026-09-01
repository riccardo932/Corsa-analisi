# Pubblicazione online — Streamlit Community Cloud

La build è pronta per essere pubblicata come app Streamlit.

## 1. Metti il progetto su GitHub

Crea un repository GitHub (può essere privato) e carica **tutti i file di questa cartella**, inclusa `.streamlit/config.toml`.

Non caricare mai una chiave API nel repository. Il file `.gitignore` esclude `.streamlit/secrets.toml`.

## 2. Crea la app su Streamlit Community Cloud

Accedi a Streamlit Community Cloud, scegli **Create app**, collega il repository GitHub e usa:

- Branch: `main`
- Main file path: `app.py`

Le dipendenze sono già elencate in `requirements.txt`.

## 3. Aggiungi il secret per l'estrazione screenshot

Nelle impostazioni avanzate / Secrets incolla:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_VISION_MODEL = "gpt-5.6-luna"
```

La chiave resta fuori dal repository. L'app funziona anche senza chiave per CSV e inserimento manuale; la chiave serve solo per l'estrazione automatica dagli screenshot.

## 4. Usa la app

1. Carica uno o più screenshot.
2. Premi **Estrai dati dagli screenshot**.
3. Controlla la tabella e correggi eventuali letture errate o duplicati.
4. Premi **Conferma dataset e rifitta da zero**.
5. Consulta modello, diagnostica, anomalie, trend, forecast e simulatori.

## Privacy e persistenza

La build non salva gli screenshot in un database. In Streamlit, i file caricati sono usati nella sessione corrente; l'estrazione vision invia l'immagine al provider API configurato. Il dataset può essere scaricato in CSV dalla app.

Questa build non include ancora autenticazione utenti o persistenza cloud multi-dispositivo. Sono componenti separati da aggiungere se vuoi trasformarla da app personale in un servizio stabile/multiutente.
