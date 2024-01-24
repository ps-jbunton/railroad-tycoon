# Railroad Tycoon (working name) Simulator

## Description
This repository contains an initial proof-of-concept simulation tool for business development operations.  It runs in Python (version > 3.3) using the package SimPy.

The simulation in `./sim/initial_attempt.ipynb` simulates the following operations:

* A set number of `ContainerYard`s are generated, each tagged with geeographical `GeoLocation` information pulled from data and a request rate parameter.
* Two `RailTerminal`s, one in Savannah and one in Cordele, are created.
* A set number of `ParallelVehicle`s are created at the Savannah terminal.
* A `Truck` is created at each `ContainerYard` and sent to the Savannah terminal to wait for `Payload`s
* We loop through the following:
    1. As container yards make requests, new `Payload`s are spawned at the Savannah terminal with designated destinations.
    2. The created Parallel vehicles load up as many `Payload`s as they can and drive them from Savannah to Cordele.
    3. The `Payload`s are delivered at Cordele, and issued to `Truck`s in FIFO order.
    4. The `Truck`s drive their `Payload`s to their destination and the `ParallelVehicle`s return to Savannah.

As of now, the results are just spewed to the console.  Next steps are to add appropriate logging/output, then hook these outputs to a visualization suite using Plotly.

To run, first set up a virtual environment with the following commands:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r ./railroad-tycoon/requirements.txt
```
Then you can load the notebook into VSCode and run the cells sequentially.