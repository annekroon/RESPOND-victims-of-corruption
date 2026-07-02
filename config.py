"""Project configuration for RESPOND corruption annotation notebooks."""

NEWS_FOLDER = "~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/"
SELECTED_COUNTRIES = ["Bulgaria", "Italy", "Netherlands", "United_Kingdom"]

COUNTRY_TO_LANG = {
    "Bulgaria": "bg",
    "Italy": "it",
    "Netherlands": "nl",
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
