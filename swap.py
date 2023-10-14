import os
import sys
import cv2
import insightface
from insightface.app import FaceAnalysis
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import shutil
import time
import moviepy.editor as mp
import argparse
import numpy as np
from gfpgan import GFPGANer
import logging
from tqdm import tqdm

VIDEO_FRAMES_DIRECTORY = "_tmp_frames"
PROCESSED_FRAMES_DIRECTORY = "_tmp_frames_out"

# Number of simultaneous processes for this script
# Tried from 1 to 8 and still the best is 2
NUMBER_OF_PROCESSES = 2

GFPGAN_MODEL_CHECKPOINT = "models/GFPGANv1.4.pth"
INSWAPPER_MODEL_CHECKPOINT = "models/inswapper_128.onnx"

total_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to the .mp4 video file to process")
parser.add_argument("face", help="Path to the image with your face")
parser.add_argument("output", help="Path to output video with .mp4 extension")
parser.add_argument("--restore", help="Enabling face restoration. Slowing down the processing", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s")
else:
    # Suppress any logging in child processes
    sys.stdout = open(os.devnull, "w")

if args.restore:
    gfpgan = GFPGANer(model_path=GFPGAN_MODEL_CHECKPOINT, upscale=1)

# Initialize FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Load the source image
source_img = cv2.imread(args.face)

# Load the swapper model
swapper = insightface.model_zoo.get_model(INSWAPPER_MODEL_CHECKPOINT, download=False, download_zip=False)
source_faces = app.get(source_img)


def process_image(image_file_name):
    target_img = cv2.imread(os.path.join(VIDEO_FRAMES_DIRECTORY, image_file_name))
    target_faces = app.get(target_img)

    res = target_img.copy()
    for face in target_faces:
        # Face swapping
        res = swapper.get(res, face, source_faces[0], paste_back=True)

        # Face restoration
        if args.restore:
            cropped_faces, restored_faces, res = gfpgan.enhance(np.array(res, dtype=np.uint8), has_aligned=False,
                                                                only_center_face=False, paste_back=True)

    output_path = os.path.join(PROCESSED_FRAMES_DIRECTORY, f"output_{image_file_name}")
    cv2.imwrite(output_path, res)


def video_to_images(video_file_path):
    list_files = []
    os.makedirs(VIDEO_FRAMES_DIRECTORY, exist_ok=True)
    cap = cv2.VideoCapture(video_file_path)
    frame_count = 0

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_skip_ratio = 1
    real_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip_ratio == 0:
            frame_filename = os.path.join(VIDEO_FRAMES_DIRECTORY, f"{frame_count:07d}.png")
            list_files.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            real_frame_count += 1

    cap.release()

    return list_files, original_fps


def images_to_video(images_path, fps, output_file):
    clip = mp.ImageSequenceClip(images_path, fps=fps)
    clip.write_videofile(output_file, fps=fps)


def get_images_list(directory):
    return [file for file in sorted(os.listdir(directory)) if
            file.endswith((".jpg", ".jpeg", ".png"))]


def remove_video_frames_directory():
    shutil.rmtree(VIDEO_FRAMES_DIRECTORY, ignore_errors=True)
    logging.info(f"Removed directory `{VIDEO_FRAMES_DIRECTORY}`")


def remove_processed_frames_directory():
    shutil.rmtree(PROCESSED_FRAMES_DIRECTORY, ignore_errors=True)
    logging.info(f"Removed directory `{PROCESSED_FRAMES_DIRECTORY}`")


if __name__ == "__main__":
    # Remove temp directories from previous run
    remove_video_frames_directory()
    remove_processed_frames_directory()

    # Start splitting
    logging.info(f"Splitting video `{args.video}` to frames")
    timer_start = time.time()
    frames, video_fps = video_to_images(args.video)
    logging.info(f"Video FPS: {video_fps}")
    logging.info(f"Total frames: {len(frames)}")
    logging.info(f"Splitting done in {time.time() - timer_start}s")

    # Create the output directory if it doesn"t exist
    os.makedirs(PROCESSED_FRAMES_DIRECTORY, exist_ok=True)

    video_frames_images_names = get_images_list(VIDEO_FRAMES_DIRECTORY)

    logging.info(f"Face restoration: {args.restore}")
    logging.info(f"Processing face swap on images using {NUMBER_OF_PROCESSES} processes")
    # Progress bar
    with tqdm(total=len(video_frames_images_names), desc="Processing Images", unit="image") as pbar:
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as executor:
            futures = {executor.submit(process_image, image_file_name) for image_file_name in video_frames_images_names}

            for completed_future in concurrent.futures.as_completed(futures):
                completed_future.result()
                pbar.update(1)

        pbar.close()

    remove_video_frames_directory()

    processed_frames_images = [os.path.join(PROCESSED_FRAMES_DIRECTORY, file) for file in get_images_list(PROCESSED_FRAMES_DIRECTORY)]
    logging.info(f"Making video from {len(processed_frames_images)} images")
    images_to_video(processed_frames_images, fps=video_fps,
                    output_file=args.output)

    remove_processed_frames_directory()

    logging.info("Video swapping completed in %s seconds" % (time.time() - total_start_time))
