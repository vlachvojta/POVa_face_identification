#!/bin/bash
# Brief: create and/or activate python environment

# Usage: ./create_or_activate_venv.sh [<env_name>]


# return $2 if $1 is empty (return default value if argument is not provided)
get_arg_or_default() {
    if ! [ -z "$1" ]; then
        echo $1
    else
        echo $2
    fi
}

ENV_NAME=$(get_arg_or_default $1 "venv")

if [ -d "$ENV_NAME" ]; then
    echo "Activating environment $ENV_NAME"
    source $ENV_NAME/bin/activate
else
    echo "Creating environment $ENV_NAME"
    python3 -m venv $ENV_NAME

    # add this line at the end of activate script: export PYTHONPATH=/path/to/your/project/root:$PYTHONPATH
    echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> $ENV_NAME/bin/activate

    echo "Envirnoment $ENV_NAME created, activating..."

    source $ENV_NAME/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi