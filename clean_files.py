import os
from PIL import Image

data_path = "C:\\Users\\tntbi\\Downloads\\data"

for root, _, files in os.walk(data_path):
    for file in files:
        try:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img.verify()
        except (IOError, SyntaxError, OSError) as e:
            print(f"Removing corrupted image: {img_path}")
            os.remove(img_path)