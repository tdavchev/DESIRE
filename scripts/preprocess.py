'''
Preprocess the dataset's annotations
'''
# #!/usr/bin/env python

import os
import csv

ROOT_DIR = "data/"
for subdir, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if "annotations.txt" in file:
            ids, frames = [], []
            xs, ys = [], []
            with open(os.path.join(subdir, file), "rb") as f:
                print(os.path.join(subdir, file))
                for line in f:
                    line = line.strip()
                    line = line.split(" ")
                    ids.append(line[0])
                    xmin = line[1]
                    ymin = line[2]
                    xmax = line[3]
                    ymax = line[4]
                    xs.append((float(xmin) + float(xmax)) / 2.0)
                    ys.append((float(ymin) + float(ymax)) / 2.0)
                    frames.append(line[5])

                f.close()
            writer = csv.writer(open(os.path.join(subdir, file[:-4])+'_processed.csv', "w"))
            writer.writerow(frames)
            writer.writerow(ids)
            writer.writerow(xs) # make sure this is correct and not reversed
            writer.writerow(ys)
