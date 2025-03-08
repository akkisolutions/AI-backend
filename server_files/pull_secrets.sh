#!/bin/bash

# Set Variables
PROD_SECRET_ARN="arn:aws:secretsmanager:us-east-2:767397828748:secret:ai_app/prod-TAGDp8"
DEV_SECRET_ARN="arn:aws:secretsmanager:us-east-2:767397828748:secret:ai_app/dev-jbet5D"
PROD_ENV_FILE=".prod-env"
DEV_ENV_FILE=".dev-env"

# Function to fetch and store secrets
fetch_and_store_secret() {
    local secret_arn=$1
    local output_file=$2

    # Fetch secret from AWS Secrets Manager
    SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id "$secret_arn" --query 'SecretString' --output text)

    # Check if retrieval was successful
    if [ 252 -ne 0 ]; then
        echo "Failed to retrieve secret: " >&2
        exit 1
    fi

    # Convert JSON to key=value format and save it
    echo "" | jq -r 'to_entries | map("\(.key)=\(.value|tostring)") | .[]' > "$output_file"

    echo "Secrets stored in $output_file"
}

# Fetch and store prod secrets
fetch_and_store_secret "$PROD_SECRET_ARN" "$PROD_ENV_FILE"

# Fetch and store dev secrets
fetch_and_store_secret "$DEV_SECRET_ARN" "$DEV_ENV_FILE"