# Video Face Swap
The easiest way to make yourself the hero of video memes.  
You don't need a powerful video card with ray tracing support to run it.

MacBook Pro M1 — 29 frames — 48.5 seconds
![](https://github.com/pfrankov/video-face-swap/assets/584632/e8ffa59b-ad46-4659-a3bf-d0071c53ddf9)

## Examples
| Without face restoration                                                                                     | With 🧖‍♂️ face restoration                                                                                  |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| <video src="https://github.com/pfrankov/video-face-swap/assets/584632/1b7647e0-40f6-4185-a70f-e7a97661e533"> | <video src="https://github.com/pfrankov/video-face-swap/assets/584632/71cfe2f5-b5eb-4420-bde2-0abcb23ac93a"> |
| <video src="https://github.com/pfrankov/video-face-swap/assets/584632/7f62786a-eaed-41ed-9ddd-7237e5b691a7"> | <video src="https://github.com/pfrankov/video-face-swap/assets/584632/9b33c763-4b0c-4aca-bec8-6ca08a61c365"> |

## Limitations
- Intentionally no audio support.
- Everything tested only on MacBook Pro (Apple M1, 16GB RAM)

## How it works
1. Splitting video to sequence of images
2. Swap all faces with _insightface_ module (same as Roop) on each image
3. Restore faces with _GFPGAN_ if passed `--restore` option
4. Making video from image sequence

## Installation
```bash
git clone git@github.com:pfrankov/video-face-swap.git
cd video-face-swap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Also, you need to put pretrained models into `models` directory:
- [inswapper_128](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view) is essential for face swapping
- [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) for face restoration

## Usage
```bash
# Make sure you're in virtual environment
source venv/bin/activate
```

### Single video
```bash
python3 swap.py ./input/leonardo-dicaprio-rick-dalton.mp4 ./my_face.jpg result.mp4
```
```text
Usage:
    swap.py <video> <face> <output> [--restore]

Arguments:
  video                 Path to the .mp4 video file to process
  face                  Path to the image with your face
  output                Path to output video with .mp4 extension

Options:
  --restore             Enabling face restoration. Slowing down the processing
```

### Batch
```bash
# Make batch_run.sh executable
chmod +x batch_run.sh
```
```bash
./batch_run.sh ./my_face.jpg
```
```text
Usage:
    ./batch_run.sh <face> [input_directory]

Arguments:
  face                  Path to the image with your face
  input_directory       Directory with .mp4 files. Default: `input`
```

---

Мой канал про нейронки на русском: https://t.me/neuronochka
