import sys
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
from PIL import Image
D=S+'ca10run2/ca10_run/top_priority_frames/'
sel=[('yolo','frame_015100_t1072.26s'),('yolo','frame_015650_t1110.65s'),('yolo','frame_015750_t1117.49s'),
     ('colour','frame_015100_t1072.26s'),('colour','frame_015600_t1107.26s'),('colour','frame_007000_t504.67s')]
fig,axs=plt.subplots(3,2,figsize=(FIG_W,FIG_W*0.88))
for i,(a,(d,st)) in enumerate(zip(axs.T.ravel(),sel)):
    im=Image.open(D+f'{d}/frames/{st}.jpg'); a.imshow(im); a.set_axis_off()
    a.text(0,1.02,f'({"abcdef"[i]}) {"YOLO11n" if d=="yolo" else "colour"}',transform=a.transAxes,fontsize=7,fontweight='bold',va='bottom')
fig.subplots_adjust(left=0.005,right=0.995,top=0.965,bottom=0.005,wspace=0.03,hspace=0.09)
fig.savefig(S+'fig/ca10/ca10_top_frames.pdf',dpi=300); fig.savefig(S+'fig/ca10/ca10_top_frames.png',dpi=170)
