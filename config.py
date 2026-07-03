"""Shared non-secret configuration for the RESPOND corpus workflow.

Secrets belong in ignored config_local.py or environment variables.
"""

import os

RD_BASE_DIR = "ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)"
NEWS_FOLDER = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/"

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

ANNOTATION_PATH = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/annotations/"
ANNOTATION_FILE = "classified_pol_corruption_validation_gabriele.csv"
ANNOTATION_ENCODING = "latin1"
VALID_CORRUPTION_LABELS = ["no political corruption", "political corruption"]

# Research Drive / Nextcloud WebDAV settings. Keep credentials out of git.
BASE_URL = os.environ.get("RD_BASE_URL", "https://uva.data.surf.nl/remote.php/dav/files")
USER = os.environ.get("RD_USER", "")
APP_PASSWORD = os.environ.get("RD_PASS")

# UvA LLM proxy settings. Keep the API key out of git.
LLMPROXY_BASE_URL = os.environ.get("LLMPROXY_BASE_URL", "https://llmproxy.uva.nl/v1")
LLMPROXY_API_KEY = os.environ.get("LLMPROXY_API_KEY")
LLMPROXY_MODEL = os.environ.get("LLMPROXY_MODEL", "gpt-5.1")

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
