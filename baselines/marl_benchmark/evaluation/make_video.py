import argparse
from pathlib import Path

import os
import shutil
import moviepy.video.io.ImageSequenceClip


def main(
        path,
):
    fps = 30

    image_files = [os.path.join(path, img)
                   for img in os.listdir(path)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(path + 'video.mp4')

    # shutil.rmtree(path + 'temp')




def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('-p',
                        '--path',
                        help='Path to images folder',
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        path=args.path,
    )
