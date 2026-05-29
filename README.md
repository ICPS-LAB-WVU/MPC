# MPC Path Tracking Controller for F1TENTH Autonomous Racing

**Author:** Mohamed Elgouhary
**Affiliation:** iCPS Lab, West Virginia University
**Project Type:** ROS 2 Autonomous Racing Controller

This repository contains a ROS 2 Python implementation of a **Model Predictive Control (MPC)** path-tracking controller for F1TENTH-style autonomous racing.

The controller follows a precomputed raceline using a kinematic vehicle model and solves a finite-horizon optimization problem to compute steering and speed commands.

## Overview

Model Predictive Control is a model-based optimal-control approach that predicts the future motion of the vehicle over a short time horizon and chooses control actions that minimize tracking error and control effort while satisfying vehicle constraints.

In this repository, MPC is used to track racing lines for F1TENTH-style autonomous driving. The controller uses the current vehicle pose, finds a local reference trajectory from the raceline, predicts the vehicle motion over a finite horizon, and publishes Ackermann steering and speed commands.

## Main Features

* ROS 2 Python implementation
* F1TENTH-compatible Ackermann drive output
* Kinematic vehicle model
* CVXPY-based optimization
* OSQP solver support
* Raceline tracking from CSV waypoint files
* Speed tracking from reference raceline velocity
* Steering, speed, and acceleration constraints
* RViz visualization of waypoints, reference trajectory, and predicted path
* Support for simulation and real-car topic configurations

## Repository Structure

```text
MPC/
├── README.md
└── src/
    ├── csv_data/
    │   ├── Austin.csv
    │   ├── Austin_fast.csv
    │   ├── Catalunya_fast.csv
    │   ├── Montreal_fast.csv
    │   ├── Monza_fast.csv
    │   ├── Spa_fast.csv
    │   ├── YasMarina_fast.csv
    │   └── ...
    └── mpc_pkg/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── resource/
        ├── test/
        └── mpc_pkg/
            ├── __init__.py
            ├── mpc.py
            └── utils.py
```

## Main Files

| File                           | Description                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| `src/mpc_pkg/mpc_pkg/mpc.py`   | Main ROS 2 MPC controller node                                     |
| `src/mpc_pkg/mpc_pkg/utils.py` | Utility functions for waypoint processing and nearest-point search |
| `src/mpc_pkg/setup.py`         | ROS 2 Python package setup file                                    |
| `src/mpc_pkg/package.xml`      | ROS 2 package metadata and dependencies                            |
| `src/csv_data/`                | Raceline CSV files for different racing tracks                     |

## Controller Description

The controller uses a kinematic state representation:

```text
x = [x_position, y_position, velocity, yaw]
```

and a control input representation:

```text
u = [acceleration, steering]
```

At each control step, the controller:

1. Receives the current vehicle pose.
2. Finds the nearest point on the raceline.
3. Builds a local reference trajectory over a finite horizon.
4. Linearizes the kinematic vehicle model.
5. Solves a constrained MPC optimization problem.
6. Publishes the first steering and speed command.
7. Visualizes the reference trajectory and predicted path in RViz.

## MPC Cost Function

The optimization balances three main objectives:

* Tracking the reference trajectory
* Penalizing large control inputs
* Penalizing rapid changes in control inputs

The controller uses cost matrices for:

| Matrix | Purpose                         |
| ------ | ------------------------------- |
| `Qk`   | State tracking error cost       |
| `Qfk`  | Final state tracking error cost |
| `Rk`   | Control input cost              |
| `Rdk`  | Control input rate cost         |

## Vehicle and MPC Parameters

Important parameters in `mpc.py` include:

| Parameter    | Description                      |
| ------------ | -------------------------------- |
| `TK`         | Prediction horizon length        |
| `DTK`        | MPC time step                    |
| `WB`         | Vehicle wheelbase                |
| `MAX_SPEED`  | Maximum vehicle speed            |
| `MIN_SPEED`  | Minimum vehicle speed            |
| `MAX_ACCEL`  | Maximum acceleration             |
| `MIN_STEER`  | Minimum steering angle           |
| `MAX_STEER`  | Maximum steering angle           |
| `MAX_DSTEER` | Maximum steering-rate constraint |

Default values include:

```text
TK = 8
DTK = 0.1
WB = 0.33
MAX_SPEED = 14.0
MAX_ACCEL = 3.0
MIN_STEER = -0.4189
MAX_STEER = 0.4189
```

## ROS 2 Topics

The controller uses the following topics by default:

| Type   | Topic                   | Description                              |
| ------ | ----------------------- | ---------------------------------------- |
| Input  | `/ego_racecar/odom`     | Vehicle odometry in simulation           |
| Input  | `/pf/viz/inferred_pose` | Estimated pose for real-car mode         |
| Output | `/drive`                | Ackermann steering and speed command     |
| Output | `/ref_traj_marker`      | RViz marker for the reference trajectory |
| Output | `/waypoints_marker`     | RViz marker for raceline waypoints       |
| Output | `/pred_path_marker`     | RViz marker for the predicted MPC path   |

## Dependencies

This package expects a ROS 2 environment with the following dependencies:

* `rclpy`
* `std_msgs`
* `sensor_msgs`
* `geometry_msgs`
* `nav_msgs`
* `ackermann_msgs`
* `visualization_msgs`
* `tf2_ros`

Python dependencies include:

* `numpy`
* `scipy`
* `cvxpy`
* `osqp`

Depending on your system, you may install Python dependencies with:

```bash
pip install numpy scipy cvxpy osqp
```

## Installation

Clone the repository into the `src` folder of your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ICPS-LAB-WVU/MPC.git
```

Build the workspace:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Running the Controller

After building and sourcing the workspace, run:

```bash
ros2 run mpc_pkg mpc
```

You should see output similar to:

```text
MPC Initialized
```

The controller will subscribe to odometry, compute MPC-based steering and speed commands, and publish them to:

```text
/drive
```

## Using a Different Track

The controller loads raceline CSV files from:

```text
src/csv_data/
```

The default map name in the code is:

```text
YasMarina_fast
```

This means the controller expects:

```text
src/csv_data/YasMarina_fast.csv
```

To use another track, modify the `map_name` variable in `mpc.py`, for example:

```python
self.map_name = 'Catalunya_fast'
```

Available track files include examples such as:

```text
Austin_fast.csv
Catalunya_fast.csv
Hockenheim_fast.csv
Melbourne_fast.csv
MexicoCity_fast.csv
Montreal_fast.csv
Monza_fast.csv
Spa_fast.csv
YasMarina_fast.csv
```

## Simulation vs Real Vehicle

The controller includes a flag for simulation or real-car operation:

```python
self.is_real = False
```

For simulation, the controller uses:

```text
/ego_racecar/odom
```

For real-car operation, the controller can use:

```text
/pf/viz/inferred_pose
```

Before running on a real vehicle, carefully verify:

* Topic names
* Vehicle coordinate frames
* Wheelbase
* Speed limits
* Steering limits
* Raceline alignment
* Odometry direction
* Emergency stop behavior

## Suggested Workflow

1. Start the F1TENTH simulator or real-car localization stack.
2. Confirm that odometry or pose messages are being published.
3. Select the desired raceline CSV file.
4. Build and source the ROS 2 workspace.
5. Run the MPC controller.
6. Visualize waypoints, reference trajectory, and predicted path in RViz.
7. Tune the MPC cost matrices and constraints if needed.
8. Test at low speed before increasing the velocity profile.

## Notes on CSV Format

The raceline CSV files are expected to contain waypoint information used by the controller, including:

* x-position
* y-position
* heading
* curvature or path-related quantities
* reference velocity

The current implementation reads waypoint columns directly from the CSV file. If you use a custom raceline, make sure the column order matches the format expected in `mpc.py`.

## Safety Notes

This controller is intended for research and development. Before using it on physical hardware:

* Test first in simulation
* Start with low speeds
* Use an emergency stop
* Confirm steering direction
* Confirm odometry direction
* Verify that the raceline is aligned with the map
* Check all ROS topic names
* Avoid running near people or obstacles during initial testing


## Author

This repository is developed and maintained by:

**Mohamed Elgouhary**
PhD Student and Graduate Research Assistant
Lane Department of Computer Science and Electrical Engineering
West Virginia University

## Acknowledgment

This repository is associated with autonomous vehicle and F1TENTH research at the iCPS Lab, West Virginia University.

## License and Attribution


The `utils.py` file includes an MIT License header and credits the original authors listed in that file. Please keep that header if you modify or redistribute the file.


## Contact

For questions or collaboration, please contact:

**Mohamed Elgouhary**
West Virginia University
Email: [mae00018@mix.wvu.edu](mailto:mae00018@mix.wvu.edu)
