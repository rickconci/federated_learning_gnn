#!/bin/bash
set -e

# Move to script's directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


python3.11 run.py --experiment_config_filename baseline_gat_experiments --experiment_name experiment_1


# # Ensure a JSON file is provided as an argument
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <path_to_json_file>"
#     exit 1
# fi

# # Assign JSON file to variable
# JSON_FILE="$1"

# # Parse JSON and run experiments
# for experiment in $(jq -r 'keys[]' "$JSON_FILE"); do
#     num_clients=$(jq -r ".$experiment.num_clients" "$JSON_FILE")
#     dataset_choice=$(jq -r ".$experiment.dataset_choice" "$JSON_FILE")
#     slice_method=$(jq -r ".$experiment.slice_method" "$JSON_FILE")
#     num_overlap=$(jq -r ".$experiment.num_overlap" "$JSON_FILE")
#     GNN_model=$(jq -r ".$experiment.GNN_model" "$JSON_FILE")
#     GNN_hidden=$(jq -r ".$experiment.GNN_hidden" "$JSON_FILE")
#     Epochs_per_client=$(jq -r ".$experiment.Epochs_per_client" "$JSON_FILE")
#     Fed_Rounds=$(jq -r ".$experiment.Fed_Rounds" "$JSON_FILE")

#     #continue with other flags
    
#     echo "Starting server"
#     python server.py \
#         --Fed_Rounds "${Fed_Rounds}" \
#         &
#     sleep 3 # Sleep for 3s to give the server enough time to start


#     echo "Running $experiment"
#     for i in $(seq 1 $num_clients); do
#         echo "Starting client $i for $experiment"
#         python client.py \
#             --num_clients "${num_clients}" \
#             --client_id "${i}" \
#             --dataset_choice "${dataset_choice}"\
#             --slice_method "${slice_method}" \
#             --num_overlap "${num_overlap}" \
#             --GNN_model "${GNN_model}" \
#             --GNN_hidden "${GNN_hidden}" \
#             --Epochs_per_client "${Epochs_per_client}" \
#              &
#     done
# done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait