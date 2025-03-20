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
if source_img is None:
    logging.error(f"Failed to load source image: {args.face}")
    sys.exit(1)

# Load the swapper model
swapper = insightface.model_zoo.get_model(INSWAPPER_MODEL_CHECKPOINT, download=False, download_zip=False)
source_faces = app.get(source_img)
if not source_faces:
    logging.error("No face detected in the source image")
    sys.exit(1)


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
    
    # Get video information using MoviePy
    video = mp.VideoFileClip(video_file_path)
    original_fps = video.fps
    original_duration = video.duration
    video.close()
    logging.info(f"Original video: FPS={original_fps}, Duration={original_duration}s")
    
    # Now extract frames using OpenCV
    cap = cv2.VideoCapture(video_file_path)
    frame_count = 0
    total_frames = 0

    frame_skip_ratio = 1
    real_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        frame_count += 1

        if frame_count % frame_skip_ratio == 0:
            frame_filename = os.path.join(VIDEO_FRAMES_DIRECTORY, f"{frame_count:07d}.png")
            list_files.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            real_frame_count += 1

    cap.release()
    
    # The effective fps should be calculated based on the number of extracted frames and original duration
    effective_fps = real_frame_count / original_duration if original_duration > 0 else original_fps
    logging.info(f"Total frames in original video: {total_frames}")
    logging.info(f"Extracted frames: {real_frame_count}")
    logging.info(f"Effective FPS for output (frames/duration): {effective_fps}")

    return list_files, effective_fps, original_duration


def extract_audio(video_path, output_path):
    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            logging.info(f"Extracting audio from {video_path} (duration: {video.duration}s)")
            audio.write_audiofile(output_path)
            audio_duration = audio.duration
            audio.close()
            video.close()
            return True, audio_duration
        else:
            logging.warning(f"No audio stream found in {video_path}")
        video.close()
        return False, 0
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        return False, 0


def images_to_video(images_path, fps, output_file, source_video=None, original_duration=None):
    try:
        # Always use FPS calculated from number of frames and original duration
        precise_fps = fps
        logging.info(f"Using effective FPS for output: {precise_fps}")
        
        # Create clip with correct FPS
        clip = mp.ImageSequenceClip(images_path, fps=precise_fps)
        clip_duration = clip.duration
        logging.info(f"Generated video clip duration: {clip_duration}s (FPS: {precise_fps})")
        
        # Check if output duration is close to original
        if original_duration and abs(clip_duration - original_duration) > 1.0:
            logging.warning(f"Output duration ({clip_duration}s) differs from original ({original_duration}s). This may cause sync issues.")
        
        if source_video:
            temp_audio = "_tmp_audio.mp3"
            has_audio, audio_duration = extract_audio(source_video, temp_audio)
            
            if has_audio:
                try:
                    audio = mp.AudioFileClip(temp_audio)
                    logging.info(f"Audio duration: {audio_duration}s")
                    
                    # If audio is longer than video, trim it
                    if audio.duration > clip.duration:
                        logging.info(f"Trimming audio to match video duration ({clip.duration}s)")
                        audio = audio.subclip(0, clip.duration)
                    
                    clip = clip.set_audio(audio)
                    logging.info(f"Writing video with audio to {output_file} (FPS: {precise_fps})")
                    clip.write_videofile(output_file, fps=precise_fps, audio_codec='aac')
                    audio.close()
                    os.remove(temp_audio)
                    logging.info(f"Video with audio saved to {output_file}")
                    return
                except Exception as e:
                    logging.error(f"Error processing audio: {str(e)}")
        
        # Fallback: write video without audio
        logging.info(f"Writing video without audio (FPS: {precise_fps})")
        clip.write_videofile(output_file, fps=precise_fps, audio=False)
        
    except Exception as e:
        logging.error(f"Error in images_to_video: {str(e)}")
        raise


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
    frames, video_fps, original_duration = video_to_images(args.video)
    logging.info(f"Video effective FPS: {video_fps}")
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
    images_to_video(processed_frames_images, 
                  fps=video_fps,
                  output_file=args.output, 
                  source_video=args.video,
                  original_duration=original_duration)

    remove_processed_frames_directory()

    logging.info("Video swapping completed in %s seconds" % (time.time() - total_start_time))
