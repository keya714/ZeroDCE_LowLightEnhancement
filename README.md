# ZeroDCE — `main.py`

Overview
-	This repository contains an implementation of Zero-DCE low-light image/video enhancement. The `main.py` script provides a simple video enhancement pipeline using the `enhance_net_nopool` model.

What `main.py` does
-	Loads a pretrained Zero-DCE model (`.pth`) and applies it frame-by-frame to an input video.
-	If the input video FPS is higher than 6, the script samples frames down to ~6 FPS (in-memory) before processing.
-	Writes an enhanced output video in MP4 format.

Requirements
-	Python 3.8+

Quick usage
-	Open `main.py` and at the bottom of the file you will find example values for `video_path`, `model_path`, and `output_path`.
-	Edit these variables to point to your input video and the saved model weights (e.g. `Epoch99.pth`).
-	Run the script:

```powershell
python main.py
```
