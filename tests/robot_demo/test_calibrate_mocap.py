"""Check stationary QTM calibration without requiring ROS."""

import unittest

from hrl_tl.robot_demo import config, pose
from scripts.robot_demo.calibrate_mocap import correction


class CalibrationTest(unittest.TestCase):
    def test_correction_places_robot_reference_at_world_origin(self) -> None:
        ros = config.RosConfig(
            heading_offset_rad=1.57,
            marker_offset_x_m=-0.011,
            marker_offset_y_m=-0.014,
        )
        samples = [
            pose.Pose2D(x=-2.0, y=0.1, yaw=-1.5, stamp=i, received_at=i)
            for i in (1.0, 2.0)
        ]
        x, y = correction(samples, ros)
        calibrated = ros.model_copy(
            update={"mocap_offset_x_m": x, "mocap_offset_y_m": y}
        )
        measured = pose.mocap_to_world(samples[0], calibrated)
        self.assertAlmostEqual(measured.x, 0.0)
        self.assertAlmostEqual(measured.y, 0.0)


if __name__ == "__main__":
    unittest.main()
