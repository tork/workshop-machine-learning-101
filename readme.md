# Machine learning 101
This workshop uses machine learning, exemplified through neural networks and [Tensorflow](https://www.tensorflow.org).

## Prerequisites
The environment has to be prepared and data downloaded on before hand.

### Clone repository
We will be running the code in Linux. Since Windows uses an extra character to denote the end of a line, Windows users should run the following command prior to cloning the repo:
`git config --global core.autocrlf false`

Git is capable of adding the additional character automatically upon cloning. That functionality is not necessary/wanted here, so we deactivate it by using the above command.

You can now clone the workshop project: `git clone https://github.com/tork/workshop-machine-learning-101.git`

### Docker
In order to make sure everyone is running compatible versions of Python and Tensorflow, we recommend using Docker and provided scripts to enter an environment prepared for running the workshop code. Note that there are two versions of Docker: A native client simply called "Docker for <your OS here>" (hereby Docker native), or Docker Toolbox. Docker Toolbox runs virtualized, while Docker native runs... natively. Since there are less moving parts using Docker native, I recommend using that if your OS supports it. Typically, Windows versions below 10 or home editions must use Toolbox, while most other can use native. Consult the Docker website for more information.

Docker Toolbox requires a virtualization driver to work. I had some issues using Windows Hyper-V in my tests. If you can't use Docker native, and are having trouble with Hyper-V, try disabling it completely before following the rest of this readme as usual.

Install Docker native or Docker Toolbox. If you run Windows, you would need to explicitly allow sharing from the volume you keep the workshop code: Right click the Docker icon in the task bar, go to settings. Under "Shared Drives", check the partition which contain the workshop project.

### environment
Open the workshop project directory and run one of the provided scripts, depending on your configuration:

OS|Docker|Script
---|---|---
Linux/macOS|Native|env-native.sh
Linux/macOS|Toolbox|env-toolbox.sh
Windows|Native|env-native.ps1
Windows|Toolbox|env-toolbox.ps1

Some versions of Windows disallows running Powershell scripts by default. If this happens to you, you need to allow running scripts. Start a Powershell session as administrator, and run the following command:
`Set-ExecutionPolicy Unrestricted`
This will allow running all scripts. I guess you would want to revert the setting once the workshop is done (`Restricted` is the default setting):
`Set-ExecutionPolicy Restricted`

`env-toolbox.ps1` has to be run as administrator (start Powershell as administrator and run the script from here). The reason is that we need admin for fetching information regarding Hyper-V.

Check that the environment container starts without issues, and that you are put in a shell. Verify that the project directory is mountet on path `/workshop-machine-learning-101`.

### Data
Data sets should be downloaded prior to solving any of the problems in this workshop. Do this by running `data.sh`. The script is runnable from the Docker container you should have running, but would also be usable with for instance macOS. Verify that data is being downloaded, and that the script prints `done` upon termination.
