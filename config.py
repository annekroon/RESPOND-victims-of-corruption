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

# UvA LLM proxy settings for translation and annotation suggestions.
# Keep the key in your shell environment or ignored config_local.py.
LLMPROXY_BASE_URL = os.environ.get("LLMPROXY_BASE_URL", "https://llmproxy.uva.nl/v1")
LLMPROXY_API_KEY = os.environ.get("LLMPROXY_API_KEY")
LLMPROXY_MODEL = os.environ.get("LLMPROXY_MODEL", "gpt-4o-mini")

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
    import config_local as _local_config

    BASE_URL = getattr(_local_config, "BASE_URL", BASE_URL) or BASE_URL
    USER = getattr(_local_config, "USER", USER) or USER
    APP_PASSWORD = getattr(_local_config, "APP_PASSWORD", APP_PASSWORD) or APP_PASSWORD
    LLMPROXY_BASE_URL = getattr(_local_config, "LLMPROXY_BASE_URL", LLMPROXY_BASE_URL) or LLMPROXY_BASE_URL
    LLMPROXY_API_KEY = getattr(_local_config, "LLMPROXY_API_KEY", LLMPROXY_API_KEY) or LLMPROXY_API_KEY
    LLMPROXY_MODEL = getattr(_local_config, "LLMPROXY_MODEL", LLMPROXY_MODEL) or LLMPROXY_MODEL
except ImportError:
    pass
