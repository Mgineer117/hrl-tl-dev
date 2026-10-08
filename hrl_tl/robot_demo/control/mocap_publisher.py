"""Qualisys motion capture ROS publisher and asynchronous client."""

from __future__ import annotations

import asyncio
import logging
import math
import xml.etree.ElementTree as element_tree
from threading import Thread
from typing import Any

import numpy as np
import qtm_rt as qtm
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node  # ty: ignore[unresolved-import]
from rclpy.qos import (  # ty: ignore[unresolved-import]
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.spatial.transform import Rotation

_LOGGER: logging.Logger = logging.getLogger(__name__)


class QualisysClient(Thread):
    """Threaded client connecting to Qualisys Track Manager (QTM)."""

    def __init__(self, ip_address: str, marker_deck_name: str) -> None:
        super().__init__(daemon=True)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.connection: qtm.QRTConnection | None = None
        self.qtm_6dof_labels: list[str] = []
        self._stay_open = True
        self.data: dict[str, Any] = {
            "time": None,
            "x": None,
            "y": None,
            "z": None,
            "yaw": None,
            "pitch": None,
            "roll": None,
        }
        self.start()

    def close(self) -> None:
        """Signal lifecycle thread to stop and wait for completion."""
        self._stay_open = False
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        """Run asyncio lifecycle loop for QTM connection."""
        asyncio.run(self._life_cycle())

    async def _life_cycle(self) -> None:
        try:
            await self._connect()
            while self._stay_open:
                await asyncio.sleep(0.5)
        except Exception:
            _LOGGER.exception("QTM client encountered an error")
        finally:
            await self._close()

    async def _connect(self) -> None:
        _LOGGER.info("Connecting to Qualisys QTM at %s", self.ip_address)
        self.connection = await qtm.connect(self.ip_address, version="1.24")
        if self.connection is None:
            raise ConnectionError(
                f"Failed to connect to QTM at {self.ip_address}"
            )
        params = await self.connection.get_parameters(parameters=["6d"])
        xml = element_tree.fromstring(params)
        self.qtm_6dof_labels = [
            label.text.strip() if label.text is not None else ""
            for label in xml.findall("*/Body/Name")
        ]
        await self.connection.stream_frames(
            components=["6d"],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet: Any) -> None:
        _, bodies = packet.get_6d()
        if bodies is None:
            return

        if self.marker_deck_name not in self.qtm_6dof_labels:
            return

        index = self.qtm_6dof_labels.index(self.marker_deck_name)
        position, orientation = bodies[index]

        t = packet.timestamp / 1e6
        x, y, z = np.array(position) / 1e3
        rot = Rotation.from_matrix(
            np.reshape(orientation.matrix, (3, -1), order="F")
        )
        yaw, pitch, roll = rot.as_euler("ZYX", degrees=False)

        self.data = {
            "time": t,
            "x": x,
            "y": y,
            "z": z,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

    async def _close(self) -> None:
        if self.connection is not None:
            try:
                await self.connection.stream_frames_stop()
                self.connection.disconnect()
            except Exception:
                _LOGGER.exception("Error while closing QTM connection")
            self.connection = None


class QualysisPublisher(Node):
    """ROS 2 node publishing Qualisys 6DoF body poses."""

    def __init__(self, ip_address: str, marker_deck_name: str) -> None:
        super().__init__("qualysis_publisher")
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.publisher_ = self.create_publisher(
            PoseStamped, f"qualysis/{marker_deck_name}", qos
        )
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.qualysis_client = QualisysClient(
            self.ip_address, self.marker_deck_name
        )

    def timer_callback(self) -> None:
        """Publish latest sample received from QualisysClient."""
        data = self.qualysis_client.data
        if data["x"] is None or data["time"] is None:
            return

        pose = PoseStamped()
        pose.header.frame_id = "mocap"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(data["x"])
        pose.pose.position.y = float(data["y"])
        pose.pose.position.z = float(data["z"])
        pose.pose.orientation.z = float(data["yaw"])
        pose.pose.orientation.y = float(data["pitch"])
        pose.pose.orientation.x = float(data["roll"])

        self.publisher_.publish(pose)


class FreshPublisher(QualysisPublisher):
    """Publish only fresh QTM samples with advancing timestamps."""

    def __init__(self, ip: str, marker: str) -> None:
        self._last_sample = float("-inf")
        super().__init__(ip, marker)

    def timer_callback(self) -> None:
        """Filter out duplicates and non-advancing timestamps."""
        data = self.qualysis_client.data
        try:
            if data["time"] is None:
                return
            sample = float(data["time"])
            valid = all(
                data[k] is not None and math.isfinite(float(data[k]))
                for k in ("x", "y", "z", "yaw", "pitch", "roll")
            )
        except (TypeError, ValueError):
            return
        if valid and math.isfinite(sample) and sample > self._last_sample:
            self._last_sample = sample
            super().timer_callback()
