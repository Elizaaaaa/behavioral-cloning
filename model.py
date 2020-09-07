import numpy as np
from scipy import ndimage

def load_images():
    import csv
    
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
