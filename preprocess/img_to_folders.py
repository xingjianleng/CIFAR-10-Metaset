from pathlib import Path
import shutil


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

path = Path("/data/lengx/cifar/cifar10-f-original/")
for sub_folder in path.iterdir():
    for class_name in CLASSES:
        (sub_folder / class_name).mkdir(exist_ok=True)
    for subsub_folder in sub_folder.iterdir():
        if subsub_folder.is_file():
            label = int(subsub_folder.name[:2])
            class_name = CLASSES[label]
            shutil.move(subsub_folder, sub_folder / class_name)
