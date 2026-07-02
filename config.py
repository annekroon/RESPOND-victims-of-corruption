"""Project configuration for RESPOND corruption annotation notebooks."""

import os

NEWS_FOLDER = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/"
RD_BASE_DIR = "ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)"

ALL_COUNTRIES = [
    "Bulgaria",
    "France",
    "Hungary",
    "Italy",
    "Netherlands",
    "Serbia",
    "Sweden",
    "Ukraine",
    "United_Kingdom",
]
SELECTED_COUNTRIES = ALL_COUNTRIES

COUNTRY_TO_LANG = {
    "Bulgaria": "bg",
    "France": "fr",
    "Hungary": "hu",
    "Italy": "it",
    "Netherlands": "nl",
    "Serbia": "sr",
    "Sweden": "sv",
    "Ukraine": "uk",
    "United_Kingdom": "en",
}

TRANSLATED_FILE = "outputs/sample_for_annotation.csv"
ANNOTATED_FILE = "outputs/sample_with_llm_suggestions.csv"
LLM_ENDPOINT = "http://localhost:11434/api/chat"
LLM_MODEL_NAME = "llama3:70b"

ANNOTATION_PATH = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/annotations/"
ANNOTATION_FILE = "classified_pol_corruption_validation_gabriele.csv"
ANNOTATION_ENCODING = "latin1"
VALID_CORRUPTION_LABELS = ["no political corruption", "political corruption"]

# Research Drive / Nextcloud WebDAV settings.
# Set credentials with environment variables or in ignored config_local.py.
BASE_URL = os.environ.get("RD_BASE_URL", "https://uva.data.surf.nl/remote.php/dav/files")
USER = os.environ.get("RD_USER", "")
APP_PASSWORD = os.environ.get("RD_PASS")

try:
    from config_local import APP_PASSWORD as LOCAL_APP_PASSWORD
    from config_local import BASE_URL as LOCAL_BASE_URL
    from config_local import USER as LOCAL_USER

    BASE_URL = LOCAL_BASE_URL or BASE_URL
    USER = LOCAL_USER or USER
    APP_PASSWORD = LOCAL_APP_PASSWORD or APP_PASSWORD
except ImportError:
    pass
