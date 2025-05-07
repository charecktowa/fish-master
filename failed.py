#!/usr/bin/env python3
"""
retry_failed_from_log.py
-----------------------
Reintenta la descarga de **solo** las imágenes que fallaron, a partir del texto
crudo de tu consola con los mensajes de «Error downloading …».

► Flujo completo
   1. Extrae todas las URLs de imagen que aparecen en el fichero de errores
      (o cualquier texto) con una expresión regular.
   2. Lee uno o varios CSV originales (los mismos que usaste para el download
      masivo) y filtra las filas cuya columna `image_url` esté en ese conjunto
      de URLs fallidas.
   3. Genera exactamente el **mismo nombre de archivo** que usa tu script
      original —cientific_name limpio + observation‑id (si existe) + índice—
      y lo guarda en el directorio de salida.
   4. Implementa reintentos con back‑off exponencial.

Ejemplo de uso
--------------
```bash
# Copia toda tu salida con errores en un fichero, p. ej. failed_errors.txt
python retry_failed_from_log.py \
       --csv minka_data/observations-417.csv minka_data/observations-416.csv \
       --error-file failed_errors.txt \
       --output-dir dataset -v
```
"""

import os
import re
import argparse
import logging
import time
from typing import Set, List

import pandas as pd
import requests
from tqdm import tqdm

# Parámetros globales ---------------------------------------------------------
RETRY_LIMIT = 5  # Nº máximo de reintentos por URL
INITIAL_BACKOFF = 3  # Segundos antes del primer reintento
TIMEOUT = 10  # Timeout por petición HTTP (s)
CHUNK = 8192  # Chunk size para la descarga en streaming
NEEDED_COLS = ["scientific_name", "image_url", "url"]

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_urls(path: str) -> Set[str]:
    """Extrae todas las URLs http/https que terminen en .jpeg, .jpg o .png."""
    pattern = re.compile(r"https?://[^\s]+\.(?:jpe?g|png)")
    urls: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            urls.update(pattern.findall(line))
    return urls


def clean_directory_name(name: str) -> str:
    if pd.isna(name):
        return "unknown"
    name = str(name).lower().strip()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w_-]", "", name)


def generate_filename(row: pd.Series, default_idx: int) -> str:
    clean_name = clean_directory_name(row["scientific_name"])
    obs_id_part = ""
    try:
        obs_id = str(row.get("url", "")).split("/")[-1]
        obs_id = re.sub(r"\W+", "", obs_id)
        if obs_id:
            obs_id_part = f"_{obs_id}"
    except Exception:  # silent fall‑back
        pass
    return f"{clean_name}{obs_id_part}_{default_idx}.jpg"


def download_with_retries(url: str, dest: str) -> bool:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(url, stream=True, timeout=TIMEOUT)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    fh.write(chunk)
            return True
        except Exception as e:
            logging.debug(f"Intento {attempt}/{RETRY_LIMIT} falló para {url}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(backoff)
                backoff *= 2
    logging.error(f"Fallo definitivo al descargar {url}")
    return False


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reintenta la descarga solo de las imágenes que fallaron, "
        "usando el texto de error y los CSV originales."
    )
    p.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Rutas de los CSV originales con metadatos",
    )
    p.add_argument(
        "--error-file",
        required=True,
        help="Archivo de texto que contiene las líneas de error",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default="dataset_retry",
        help="Directorio donde guardar las imágenes",
    )
    p.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.15,
        help="Pausa entre descargas, en segundos (def. 0.15)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Salidas de depuración")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    os.makedirs(args.output_dir, exist_ok=True)

    failed_urls = extract_urls(args.error_file)
    logging.info(f"URLs extraídas del log: {len(failed_urls)}")
    if not failed_urls:
        logging.error("No se encontraron URLs en el archivo de error.")
        return

    # Carga y concatena todos los CSV
    frames: List[pd.DataFrame] = []
    for csv_path in args.csv:
        logging.debug(f"Leyendo {csv_path}…")
        frames.append(
            pd.read_csv(csv_path, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
        )
    data = pd.concat(frames, ignore_index=True)

    subset = data[data["image_url"].isin(failed_urls)]
    logging.info(f"Filas encontradas en CSV: {len(subset)}")
    if subset.empty:
        logging.error("Ninguna de las URLs fallidas aparece en los CSV.")
        return

    success = error = 0
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Retrying"):
        filename = generate_filename(row, idx)
        dest = os.path.join(args.output_dir, filename)
        if os.path.exists(dest):
            continue  # ya descargada
        if download_with_retries(row["image_url"], dest):
            success += 1
        else:
            error += 1
        time.sleep(args.delay)

    logging.info(f"Descargas completadas: {success}  |  Fallos: {error}")


if __name__ == "__main__":
    main()
