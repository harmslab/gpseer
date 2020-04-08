import pathlib
from urllib.parse import urljoin
import requests
from tqdm import tqdm


def main(
    logger,
    output_dir="examples"
):
    """Fetch the GPSeer examples directory.
    """
    p = pathlib.Path(output_dir)
    p.mkdir(exist_ok=True)

    files_to_download = {"example-full.csv", "example-train.csv", "example-test.csv"}

    logger.info(f"Downloading files to /{output_dir}...")
    for fname in tqdm(files_to_download, desc="[GPSeer] └──>"):
        base_url = "https://raw.githubusercontent.com/harmslab/gpseer/master/examples"
        r = requests.get("/".join([base_url, fname]))
        (p / fname).write_text(r.text)
    logger.info("└──> Done!")
