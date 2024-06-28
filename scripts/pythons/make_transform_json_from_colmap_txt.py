""" Ref. https://github.com/colmap/colmap/blob/51dc6092d1329615696d3ba60e30f2b7ce475693/scripts/python/read_write_model.py
LICENSE
# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
from pathlib import Path
import numpy as np
import collections
import json
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    return parser.parse_args()


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "BaseImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def read_cameras_text(path: str) -> dict:
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_images_text(path: str) -> dict:
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir

    colmap_dir = target_dir / "colmap_sparse/rig_txt"

    cameras_path = colmap_dir / "cameras.txt"
    images_path = colmap_dir / "images.txt"
    points3D_path = colmap_dir / "points3D.txt"

    cameras: dict[int, Camera] = read_cameras_text(str(cameras_path))
    images: dict[int, Image] = read_images_text(str(images_path))

    json_frames = []
    for image_id, image in images.items():
        camera_name = image.name.split("/")[0]
        filename = image.name.split("/")[-1]
        timestamp = int(filename.replace(".jpeg", ""))

        camera = cameras[image.camera_id]

        r = Rotation.from_quat(image.qvec)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r.as_matrix()
        transform_matrix[:3, 3] = image.tvec

        json_frames.append(
            {
                "file_path": f"images/{image.name}",
                "fl_x": camera.params[0],
                "fl_y": camera.params[1],
                "cx": camera.params[2],
                "cy": camera.params[3],
                "w": int(camera.width),
                "h": int(camera.height),
                "camera_model": "OPENCV",
                "camera": camera_name,
                "timestamp": float(timestamp / 1e9),
                "k1": camera.params[4],
                "k2": camera.params[5],
                "k3": 0.0,
                "k4": 0.0,
                "p1": camera.params[6],
                "p2": camera.params[7],
                "transform_matrix": transform_matrix.tolist(),
            }
        )

    json_data = {"frames": json_frames}
    json_path = target_dir / "transform.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    annotation_json: dict[str, list] = {"frames": []}
    annotation_json_path = target_dir / "annotation.json"
    with open(annotation_json_path, "w") as f:
        json.dump(annotation_json, f, indent=4)
