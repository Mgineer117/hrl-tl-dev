# Robot Demo

## Objective
We want to add robot demo using TurtleBot for one of the envs (ZoneEnv) for realtime inference of the trained hierarchical policy.

## Implementation Overivew
We use the mocap system to get the current position of the Turtlebot.
The state estimation from the mocap system is not the same as the env representation, so we have to create a wrapper to convert the state estimation to the env representation.


## Simulator
We will use Zone env.
The inference code on sim is available in [scripts/cpc/main_tl_hrl_zone_rep_sdsac.py](../scripts/cpc/main_tl_hrl_zone_rep_sdsac.py).

## Robot Setup
An example code for running the policy on a real robot is available at [LIRA-illinois/mrs2025](https://github.com/LIRA-illinois/mrs2025).

You can ignore the [MARL part](https://github.com/LIRA-illinois/mrs2025/tree/main/mrs2025/marl) of the codebase; it is for a MARL communication project, which is not relevant to us.
[`README.md`](https://github.com/LIRA-illinois/mrs2025/blob/main/README.md) contains the procedure to run Turtlebots.

## Gitflow
The recommended gitflow is to create a feature branch (e.g., `feat/demo`) from the `dev` branch for this project repo (`hrl_tl`) and work on it.
You can create a dedicated directoris (e.g., `hrl_tl/robot_demo`, `scripts/robot_demo`, etc.) to store files/modules.

# Coding
Make sure we follow the [conding best-practices](https://github.com/LIRA-illinois/coding_skills/blob/main/skills/python/SKILL.md) and [CONTRIBUTING.md](../CONTRIBUTING.md)

For agentic-coding, if you are using VS Code, recommendation is to add all the relevant repos (e.g., `hrl_tl` and `mrs2025`) to a workspace so that the agent can access the files.