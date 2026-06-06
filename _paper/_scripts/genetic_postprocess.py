import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from file_operations import read_from_csv
import matplotlib.pyplot as plt
from geometry.geometry_fourier import decode_genome
from launch_geometry import geometry_construct
import config as cfg
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import glob

rows = read_from_csv("./results/Type2_genetic.csv")

def generate_files():
    for i, r in enumerate(rows):
        print(i)
        toroid, poloid = decode_genome(np.array(r))
        geometry_construct(toroid,
                        poloid,
                        init=True, 
                        plot=True,
                            filename=f"final_geometry_gif_00{i}")

def pdfs_to_gif(pdf_folder, output_gif="DONUT_genetic.gif", fps=8):
    images = []

    pdf_files = sorted(glob.glob(f"{pdf_folder}/*_gif_*.pdf"))

    for pdf in pdf_files:
        pages = convert_from_path(pdf)

        # take first page only (or loop if multi-page PDFs)
        img = pages[0].convert("RGB")
        images.append(img)

    # save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0
    )

def plot_history(filename):
    history = np.array(read_from_csv(filename))
    plt.plot(history[:,0], color = 'k')
    plt.ylabel('Objective')
    plt.xlabel('Generations')
    plt.grid(True)
    plt.savefig("./_paper/loss.pdf")

#generate_files()
#pdfs_to_gif("./results/")
plot_history("./results/Type2_genetic_history.csv")