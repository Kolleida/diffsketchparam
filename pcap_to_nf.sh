#!/usr/bin/env bash

# Runs CICFlowMeter to convert pcap files to Netflow format.

trap "exit 130" SIGINT

INITIAL_WORKING_DIR="$(pwd)"

CFM_PATH="CICFlowMeter/build/distributions/CICFlowMeter-4.0/bin/cfm"
FILE_REGEX='(.*).pcap'

pcap_path="/path/to/*.pcap" # Where the pcap files are located.
output_dir="/path/to/output_directory" # Where to save the converted Netflow CSV files.
error_log="cfm_error.log"

start_time=$(date +%s)
echo "START: $(date)"

for pcap_file in $pcap_path; do
    file_name=$(basename "${pcap_file%.pcap}")

    # Check if CSV for file already exists.
    output_path="${output_dir}/${file_name}.pcap_Flow.csv"
    if [[ -f $output_path ]]; then
        echo "Skipping ${file_name}, CSV already exists."
        continue
    fi

    echo "Processing: ${file_name}"
    # Change to the directory of the CFM script (some configurations use relative paths). Remember to change back after running/upon error.
    cd "$(dirname "$CFM_PATH")"
    $CFM_PATH "${pcap_file}" "${output_dir}" 2>>$error_log > /dev/null || {
        echo "Error processing ${file_name}. Skipping."
        cd "$INITIAL_WORKING_DIR"
        rm -f "$output_path"  # Remove incomplete output file
        continue
    }
    cd "$INITIAL_WORKING_DIR"

done

end_time=$(date +%s)
echo "END: $(date)"
echo "Total time taken: $(($end_time - $start_time)) seconds"