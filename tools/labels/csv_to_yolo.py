"""makesense.ai CSV export -> YOLO txt (one per image, empty if no boxes). Usage: python3 csv_to_yolo.py test_C1"""
import csv, glob, os, sys
CLS = {'weed': 0, 'trunk': 1, 'weed_cluster': 2, 'weed cluster': 2}
s = sys.argv[1]; f = sorted(glob.glob(f'{s}/labels/labels_*.csv'))[-1]   # newest export
lab = {}
for r in csv.DictReader(open(f)):
    x, y, w, h, W, H = (float(r[k]) for k in ('bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_width', 'image_height'))
    lab.setdefault(r['image_name'], []).append(f"{CLS[r['label_name'].strip()]} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
n = 0
for img in sorted(os.listdir(f'{s}/images')):
    if not img.endswith('.jpg'): continue
    open(f'{s}/labels/{img[:-4]}.txt', 'w').write('\n'.join(lab.get(img, [])) + ('\n' if img in lab else '')); n += 1
print(f'{s}: {os.path.basename(f)} -> {n} txt files, {sum(map(len, lab.values()))} boxes')
