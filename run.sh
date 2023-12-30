#!/bin/bash
set -e

# Move to script's directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Ensure a JSON file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_json_file>"
    exit 1
fi


# Assign JSON file to variable
JSON_FILE="$1"


echo "Starting server"
python server.py &
sleep 3 # Sleep for 3s to give the server enough time to start

# Parse JSON and run experiments
for experiment in $(jq -r 'keys[]' "$JSON_FILE"); do
    num_clients=$(jq -r ".$experiment.num_clients" "$JSON_FILE")
    num_overlap=$(jq -r ".$experiment.num_overlap" "$JSON_FILE")
    
    echo "Running $experiment with $num_clients clients and $num_overlap overlap"
    for i in $(seq 1 $num_clients); do
        echo "Starting client $i for $experiment"
        python client.py --node-id "${i}" --num_overlap "${num_overlap}" &
    done
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait