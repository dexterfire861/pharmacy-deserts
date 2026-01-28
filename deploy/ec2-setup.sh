#!/bin/bash
# EC2 Instance Setup Script for Pharmacy Desert Explorer
# Run this script on a fresh Amazon Linux 2023 or Ubuntu 22.04 EC2 instance
#
# Usage:
#   chmod +x ec2-setup.sh
#   sudo ./ec2-setup.sh

set -e

echo "=========================================="
echo "Pharmacy Desert Explorer - EC2 Setup"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
fi

echo "Detected OS: $OS"

# Update system
echo "Updating system packages..."
if [[ "$OS" == *"Amazon"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    sudo yum update -y
    sudo yum install -y docker git
elif [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose git
fi

# Start and enable Docker
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add ec2-user to docker group
echo "Adding user to docker group..."
if id "ec2-user" &>/dev/null; then
    sudo usermod -aG docker ec2-user
elif id "ubuntu" &>/dev/null; then
    sudo usermod -aG docker ubuntu
fi

# Install Docker Compose (if not already installed)
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create app directory
echo "Creating application directory..."
sudo mkdir -p /opt/pharmacy-deserts
sudo chown $(whoami):$(whoami) /opt/pharmacy-deserts

# Clone or pull repository (update URL as needed)
echo "Setting up application..."
cd /opt/pharmacy-deserts

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << 'EOF'
# Pharmacy Desert Explorer Configuration
ENVIRONMENT=production
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
REQUIRE_AUTH=true
APP_PASSWORD=change-this-password
DATA_DIR=raw_data
RESULTS_DIR=results
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ADDRESS=0.0.0.0
EOF
    echo "IMPORTANT: Edit /opt/pharmacy-deserts/.env with your configuration!"
fi

# Create systemd service for auto-restart
echo "Creating systemd service..."
sudo cat > /etc/systemd/system/pharmacy-deserts.service << 'EOF'
[Unit]
Description=Pharmacy Desert Explorer
Requires=docker.service
After=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
WorkingDirectory=/opt/pharmacy-deserts
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable pharmacy-deserts

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your application code to /opt/pharmacy-deserts/"
echo "2. Edit /opt/pharmacy-deserts/.env with your configuration"
echo "3. Upload your data files to your S3 bucket"
echo "4. Run: cd /opt/pharmacy-deserts && docker-compose up -d"
echo ""
echo "Or start the service with:"
echo "   sudo systemctl start pharmacy-deserts"
echo ""
echo "View logs with:"
echo "   sudo journalctl -u pharmacy-deserts -f"
echo ""

