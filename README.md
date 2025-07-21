# Inverted Pendulum Control

This project demonstrates the control of an inverted pendulum using different control methods within the `gym` framework. The goal is to stabilize the pendulum using various controllers such as LQR (Linear Quadratic Regulator) and MPC (Model Predictive Control).

## Setup

To get started with this project, clone this repository to your local machine:

```
git clone https://github.com/JiatongSun/inverted_pendulum.git
cd inverted_pendulum
```

## Requirements
Make sure to install the necessary dependencies. Virtual environment is strongly recommended.
You can use pip to install them:

```
pip install -r requirements.txt
```

## Usage
The core functionality of the project is implemented in the `pendulum_env.py` file. The `run_pendulum_env` function in the main function allows you to select different control methods by setting the `control_method` parameter.

## Control Methods
Currently, the following control methods are supported:

**LQR (Linear Quadratic Regulator):** `"lqr"`

**MPC (Model Predictive Control):** `"mpc"`

## Running the Simulation
To run the simulation with a chosen controller, modify the control_method parameter in the main function of pendulum_env.py. For example:

```
def main():
    run_pendulum_env(control_method="lqr")
```
Then, run the script:
```
python pendulum_env.py
```
This will launch the simulation with the selected controller.

## Contributing
If you'd like to contribute to the project, feel free to fork the repository, make changes, and create a pull request. Any suggestions or improvements are welcome!
