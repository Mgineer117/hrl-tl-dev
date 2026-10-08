"""Run the bundled TurtleBot demo from the repository root."""

from absl import app

from scripts.robot_demo.run import main


if __name__ == "__main__":
    app.run(main)
