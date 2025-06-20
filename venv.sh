#!/bin/bash

# This script is used for creating the virtual env, activating it, detactivating it and removing it by using a simple bash script.
# Set default venv name
VENV=".ds_covid19"
# Get working directory
PWD=$(pwd)

# Create python virtual environment
function create_venv(){
    # Check if the folder for venv exsts, if not, create the env, if not activate the env.
    if [ -d $VENV ]; then
        echo "$VENV already exists"
        activate_venv
    else
        python3 -m venv $VENV
    fi
}

# Activate existing python virtual environment
function activate_venv(){
    . $PWD/$VENV/bin/activate
    echo "Environment '$VENV' activated"
}

# Deactivate existing python virtual environment
function deactivate_venv(){
    deactivate
}

# Clean existing python virtual environment
function clean_venv(){
    rm -rf $VENV
}

# Show available argument for this script
function help(){
    echo "usage: <create|activate|deactivate|clear|help>
        Commands:
            create: creates a new virtual envrinoment using a fixed name.
            activate: activates an existing virtual envrinoment.
            deactivate: deactivates an existing virtual envrinoment.
            clear: removes an existing virtual envrinoment.
            help: shows the available commands with examples.

        Examples:
        source venv.sh create"
}


if [ "$1" = "help" ]; then
    help
elif [ "$1" = "create" ]; then
    create_venv
elif [ "$1" = "activate" ]; then
    activate_venv
elif [ "$1" = "deactivate" ]; then
    deactivate_venv
elif [ "$1" = "clean" ]; then
    clean_venv
else
    echo "'$1' command not found, please use 'help' to find the options for this script!"
fi
