# Set the k2eg config
set -e

CONFIG_DIR="/app/src/config"
CONFIG_FILE="$CONFIG_DIR/lcls.ini"

# Check for required environment variables
if [ -z "$K2EG_IP" ]; then
  echo "Error: K2EG_IP environment variable is not set." >&2
  exit 1
fi
if [ -z "$K2EG_TOPIC" ]; then
  echo "Error: K2EG_TOPIC environment variable is not set." >&2
  exit 1
fi

# Ensure config directory exists
mkdir -p "$CONFIG_DIR"

# Write the .ini file
cat > "$CONFIG_FILE" <<EOF
[DEFAULT]

kafka_broker_url=$K2EG_IP

k2eg_cmd_topic=$K2EG_TOPIC
EOF

echo "Wrote k2eg config to $CONFIG_FILE"
