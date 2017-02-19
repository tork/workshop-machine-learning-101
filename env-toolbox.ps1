#Requires -RunAsAdministrator

param([string]$driver="")

# assert that docker-machine is working
docker-machine --version >$null 2>&1
if (!$?) {
    echo "Failed to run docker-machine --version. Is Docker Toolbox properly installed?"
    exit
}

$MACHINE="ml101"
# check if workshop machine doesn't exist
docker-machine inspect $MACHINE >$null 2>&1
if (!$?) {
    echo "Creating docker machine $MACHINE"
    $VDRIVER=$driver
    if ($VDRIVER -eq "") {
        $VDRIVER="virtualbox"

        # use hyper-v driver if available
        $HYPERV=Get-WindowsOptionalFeature -FeatureName Microsoft-Hyper-V-All -Online
        if (!$?) {
            echo "Failed to get hyper-v status, defaulting to $VDRIVER"
        } elseif ($HYPERV.State -eq "Enabled") {
            $VDRIVER="hyperv"
        }
    }

    # create workshop machine
    docker-machine create --driver $VDRIVER $MACHINE
}

# load workshop machine environment variables
docker-machine env $MACHINE | Invoke-Expression

# run workshop image
& $PSScriptRoot\env-native.ps1