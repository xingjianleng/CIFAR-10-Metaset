import logging
import mimetypes
import os
import requests
import sys
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
HEADERS = {"User-Agent": "Mozilla/5.0"}
CLASSES = (
    "airplane",
    "sedan",
    "suv",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    label = sys.argv[1]
    assert label in CLASSES

    urls = []
    with open(f"{label}.txt", "r") as f:
        line = f.readline()
        while line:
            urls.append(line[10: -4])
            line = f.readline()

    target_path = f"{label}/"
    logging.basicConfig(filename=f"{label}.log", encoding="utf-8", level=logging.ERROR)

    try:
        # make the target path
        os.mkdir(target_path)
    except OSError:
        pass

    for i, url in enumerate(tqdm(urls)):
        if url.startswith("x-raw-image:///"):
            # skip Google x-raw-image
            continue
        try:
            # disable the verification in case their certificate expired
            response = requests.get(url, headers=HEADERS, verify=False, allow_redirects=True, timeout=20)
            content_type = response.headers["Content-Type"].split(";")[0]
            if "html" in content_type:
                # skip html files
                continue
            # guess the file extension from the Content-Type
            guessed_extension = mimetypes.guess_extension(content_type)
            if guessed_extension is None:
                # store `octet-stream`, .jpg and .jpeg as .jpg
                guessed_extension = ".jpg"
            with open(f"{target_path}{i}{guessed_extension}", "wb") as f:
                f.write(response.content)
        except Exception:
            # fail to load the webpage
            logging.exception(f"Request timeout, process manually\nurl: {url}", exc_info=False)
