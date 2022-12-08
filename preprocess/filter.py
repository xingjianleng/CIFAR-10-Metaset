from PIL import Image
from pathlib import Path


def main():
	path = Path("~/Downloads/dataset/G-12/ship").expanduser().absolute()
	for file in path.iterdir():
		suffix = file.suffix
		if suffix == ".html" or suffix == ".svg" or suffix == "ico":
			file.unlink()
		elif suffix != ".jpg" and not file.name == ".DS_Store":
			try:
				img = Image.open(file).convert("RGB")
				img.save(path / (file.stem + ".jpg"), "jpeg")
				file.unlink()
			except:
				# file might be broken
				file.unlink()


if __name__ == "__main__":
	main()
