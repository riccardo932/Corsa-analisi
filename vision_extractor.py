from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from data_utils import CANONICAL_COLUMNS, canonicalize_dataframe

DEFAULT_VISION_MODEL = "gpt-5.6-luna"

RUN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": ["string", "null"]},
                    "duration_min": {"type": ["number", "string", "null"]},
                    "distance_km": {"type": ["number", "string", "null"]},
                    "pace_min_km": {"type": ["number", "string", "null"]},
                    "avg_hr_bpm": {"type": ["number", "string", "null"]},
                    "elevation_m": {"type": ["number", "string", "null"]},
                    "power_w": {"type": ["number", "string", "null"]},
                    "cadence_spm": {"type": ["number", "string", "null"]},
                    "rpe": {"type": ["number", "string", "null"]},
                },
                "required": [
                    "date", "duration_min", "distance_km", "pace_min_km", "avg_hr_bpm",
                    "elevation_m", "power_w", "cadence_spm", "rpe"
                ],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["runs", "notes"],
    "additionalProperties": False,
}

EXTRACTION_INSTRUCTIONS = """
Sei un estrattore rigoroso di dati da screenshot di attività di corsa (Strava, Garmin,
Apple Fitness e app simili). Estrai una riga per ogni corsa chiaramente rappresentata.

Campi: data, durata, distanza, passo medio, FC media, dislivello positivo, potenza media,
cadenza media, RPE.

Regole obbligatorie:
- Non inventare, stimare o inferire valori non visibili.
- Se un campo non è chiaramente leggibile, restituisci null.
- Non calcolare il passo da durata/distanza e non calcolare durata da passo/distanza.
- Non confondere FC massima con FC media, né dislivello con quota.
- Mantieni le unità semantiche richieste: km, min/km, bpm, metri, watt, spm.
- Per il passo usa preferibilmente una stringa M:SS quando visibile.
- Per la durata puoi restituire HH:MM:SS o MM:SS se mostrata.
- La data deve essere quella dell'attività; se manca l'anno o la data è ambigua, usa null e spiega nelle notes.
- Se lo screenshot mostra split/lap, non trattarli come corse separate salvo evidenza esplicita.
- Nelle notes segnala ambiguità, testo tagliato o campi che potrebbero essere letti male.
""".strip()


def _data_url(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_api_key(explicit_key: str | None = None) -> str | None:
    return explicit_key or os.getenv("OPENAI_API_KEY")


def extract_runs_from_image(
    file_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract canonical run fields from one screenshot using OpenAI vision.

    Returns only fields explicitly extracted from the image. No deterministic derivation
    is performed here; canonicalization only converts formats/units.
    """
    if not api_key:
        raise ValueError("Manca OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Pacchetto openai non installato. Esegui pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": EXTRACTION_INSTRUCTIONS},
                    {"type": "input_image", "image_url": _data_url(file_bytes, mime_type)},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "running_activity_extraction",
                "schema": RUN_SCHEMA,
                "strict": True,
            }
        },
    )
    raw = response.output_text
    payload = json.loads(raw)
    raw_df = pd.DataFrame(payload.get("runs", []))
    if raw_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), payload.get("notes", [])
    # canonicalize does not derive any missing metric.
    return canonicalize_dataframe(raw_df, derive_pace=False), payload.get("notes", [])
