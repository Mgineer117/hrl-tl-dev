"""Publishes fresh QTM rigid-body poses over ROS 2."""

from __future__ import annotations

import collections.abc
import os

import rclpy  # ty: ignore[unresolved-import]
from absl import app, flags

from hrl_tl.robot_demo.control import mocap_publisher

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "128.174.245.64", "QTM server IP address.")
flags.DEFINE_string("marker", "tb3_1", "QTM rigid-body name.")


def main(argv: collections.abc.Sequence[str]) -> None:
    """CLI entry point for publishing motion capture poses."""
    del argv  # Unused.
    if os.environ.get("ROS_DOMAIN_ID") != "40":
        raise app.UsageError(
            "Set ROS_DOMAIN_ID=40 before publishing mocap poses."
        )

    rclpy.init(args=[])
    publisher = mocap_publisher.FreshPublisher(FLAGS.ip, FLAGS.marker)
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.qualysis_client.close()
        publisher.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    app.run(main)
