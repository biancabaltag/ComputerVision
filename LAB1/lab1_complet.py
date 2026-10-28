
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "rezultate"
OUT.mkdir(exist_ok=True)


def read_image(filename):
    path = BASE / filename
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {path}")
    return img


def save_image(filename, image):
    path = OUT / filename
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Nu pot salva {path}")
    return path


def comparison(filename, entries, columns=3):
   
    rows = (len(entries) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, image) in zip(axes, entries):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=140)
    plt.close(fig)


def special_filter(image, a):
    if not 0 <= a <= 2:
        raise ValueError("a trebuie sa fie intre 0 si 2")
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8 + a, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def rotate_image(image, angle):

    
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_width = int(np.ceil(height * sin_a + width * cos_a))
    new_height = int(np.ceil(height * cos_a + width * sin_a))
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        image, matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def crop_image(image, x, y, width, height):
    img_height, img_width = image.shape[:2]
    if not all(isinstance(v, int) for v in (x, y, width, height)):
        raise TypeError("Coordonatele si dimensiunile trebuie sa fie intregi")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Coordonate sau dimensiuni invalide")
    if x + width > img_width or y + height > img_height:
        raise ValueError("Zona de crop depaseste imaginea")
    return image[y:y + height, x:x + width].copy()


def make_emoji():
    emoji = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.circle(emoji, (200, 200), 155, (0, 190, 255), -1)
    cv2.circle(emoji, (200, 200), 155, (0, 120, 225), 5)
    cv2.ellipse(emoji, (145, 155), (13, 23), 0, 0, 360, (35, 35, 35), -1)
    cv2.ellipse(emoji, (255, 155), (13, 23), 0, 0, 360, (35, 35, 35), -1)
    cv2.ellipse(emoji, (200, 220), (90, 65), 0, 10, 170, (35, 35, 35), 9)
    cv2.circle(emoji, (110, 220), 16, (140, 175, 255), -1)
    cv2.circle(emoji, (290, 220), 16, (140, 175, 255), -1)
    return emoji


def main():
    print("Exercitiul 1 - OpenCV", cv2.__version__)
    img = read_image("lena.tif")

    # Ex 2 
    height, width, channels = img.shape
    print(f"Exercitiul 2 - latime={width}, inaltime={height}, canale={channels}")
    save_image("02_original.jpg", img)
    comparison("02_afisare_original.png", [("Imagine originala", img)], 1)

    # Ex 3 
    results = [
        ("Original", img),
        ("Average 3x3", cv2.blur(img, (3, 3))),
        ("Average 9x9", cv2.blur(img, (9, 9))),
        ("Gaussian 3x3", cv2.GaussianBlur(img, (3, 3), 0)),
        ("Gaussian 11x11", cv2.GaussianBlur(img, (11, 11), 0)),
        ("Median 3", cv2.medianBlur(img, 3)),
        ("Median 9", cv2.medianBlur(img, 9)),
    ]
    for strength in (1, 2):
       
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1 + 4 * strength, -strength],
            [0, -strength, 0]
        ], dtype=np.float32)
        results.append((f"Sharpen {strength}", cv2.filter2D(img, -1, kernel)))
    for index, (title, result) in enumerate(results[1:], start=1):
        save_image(f"03_filtru_{index:02d}.png", result)
    comparison("03_comparatie_filtre.png", results, 3)
    print("Exercitiul 3 - filtre salvate")

    # Ex 4 - 
    files = [("lena", img)]
    second_path = BASE / "imagine2.jpg"
    if second_path.exists():
        files.append(("imagine2", read_image("imagine2.jpg")))
    else:
        print("ATENTIE: adauga imagine2.jpg pentru a termina cerinta ex. 4!")
    for label, source in files:
        compare = [("Original", source)]
        for a in (0, 0.5, 1, 1.5, 2):
            filtered = special_filter(source, a)
            save_image(f"04_{label}_a_{str(a).replace('.', '_')}.png", filtered)
            compare.append((f"a = {a}", filtered))
        comparison(f"04_{label}_comparatie.png", compare, 3)
    print("Exercitiul 4 - filtru special salvat")

    # Ex 5 
    rotation_results = [("Original", img)]
    for angle in (30, -30, 45, -60, 90):
        rotated = rotate_image(img, angle)
        save_image(f"05_rotire_{angle}.png", rotated)
        rotation_results.append((f"{angle} grade", rotated))
    comparison("05_comparatie_rotiri.png", rotation_results, 3)
    print("Exercitiul 5 - rotiri salvate")

    # Ex 6 
    x, y = width // 4, height // 4
    crop_width, crop_height = width // 2, height // 2
    cropped = crop_image(img, x, y, crop_width, crop_height)
    save_image("06_crop.png", cropped)
    comparison("06_comparatie_crop.png", [("Original", img), ("Crop", cropped)], 2)
    print(f"Exercitiul 6 - crop: ({x}, {y}), {crop_width}x{crop_height}")

    # Ex 7 
    emoji = make_emoji()
    save_image("bianca_baltag.jpg", emoji)
    print("Exercitiul 7 - emoji salvat: bianca_baltag.jpg")
    print(f"GATA! Toate rezultatele sunt in: {OUT}")


if __name__ == "__main__":
    main()
