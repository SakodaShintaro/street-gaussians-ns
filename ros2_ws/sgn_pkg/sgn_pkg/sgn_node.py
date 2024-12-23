import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path

import torch
if torch.__version__.split("+")[0] >= "2.1.0":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

from typing import Callable, Literal, Optional, Tuple

import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    print(config)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    return config, pipeline


class SgnNode(Node):
    def __init__(self):
        super().__init__('sgn_node')

        self.bridge = CvBridge()
        self.image_publishers = {}
        self.camera_transforms = []

        image_topics = [
            "/sensing/camera/camera0/image_rect_color/compressed",
            "/sensing/camera/camera1/image_rect_color/compressed"
        ]
        self.camera_transforms = [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        ]

        if len(image_topics) != len(self.camera_transforms):
            self.get_logger().error("The number of image topics and camera transforms must match.")
            return

        for image_topic in image_topics:
            self.image_publishers[image_topic] = self.create_publisher(Image, image_topic, 10)

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/localization/pose_estimator/pose',
            self.pose_callback,
            10
        )

        self.get_logger().info(f"Publishing to: {image_topics}")

        self.config_path = Path("/home/user/data/rosbag/20241221_for_3dgs/output/street-gaussians-ns/street-gaussians-ns/2024-12-23_130727/config.yml")
        config, pipeline, _, _ = eval_setup(
            self.config_path,
            eval_num_rays_per_chunk=1024,
            test_mode="inference",
        )
        self.pipeline = pipeline

    def pose_callback(self, base_pose):
        for image_topic, transform in zip(self.image_publishers.keys(), self.camera_transforms):
            camera_pose = self.transform_pose(base_pose.pose, transform)
            image = self.render_image(camera_pose)
            self.publish_image(self.image_publishers[image_topic], image)

    def transform_pose(self, pose, transform):
        """Transform the pose using a 4x4 matrix."""
        return pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z, 1])
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Apply the transformation to position
        transformed_position = np.dot(transform, position)

        # Apply the transformation to orientation
        rotation_matrix = transform[:3, :3]
        quaternion = tf_transformations.quaternion_from_matrix(
            np.block([
                [rotation_matrix, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))]
            ])
        )
        return Pose(
            position=transformed_position[:3],
            orientation=quaternion
        )

    def render_image(self, pose):
        """Render an image based on the transformed pose."""
        with torch.no_grad():
            # Prepare the camera parameters for rendering
            camera = {
                "position": torch.tensor(pose, dtype=torch.float32),
                "orientation": torch.eye(3, dtype=torch.float32),
            }
            outputs = self.pipeline.model.get_outputs_for_camera(camera)

        # Extract RGB output
        output_image = outputs["rgb"].cpu().numpy()

        # Convert to valid image format
        output_image = (output_image * 255).astype(np.uint8)
        return output_image

    def publish_image(self, publisher, cv_image):
        image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        publisher.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SgnNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
