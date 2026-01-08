#!/bin/bash
set -e  # exit immediately on error

echo "Starting SAM2 Backend Server..."
echo ""
echo "Requirements:"
echo "1. pip install -r requirements.txt"
echo "2. HF_TOKEN environment variable (optional, for private repos)"
echo ""

# Resolve script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PUBLIC_DIR="$SCRIPT_DIR/public"

# # Create directories if they don't exist
mkdir -p "$PUBLIC_DIR"

# echo "Model directory: $MODEL_DIR"
echo "Public directory: $PUBLIC_DIR"
echo ""

# HuggingFace auth header (only if token exists)
AUTH_HEADER=""
if [[ -n "$HF_TOKEN" ]]; then
  AUTH_HEADER="Authorization: Bearer $HF_TOKEN"
fi

echo "Downloading SAM2 encoder..."


echo "Downloading SAM2 decoder..."

curl -L \
  ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
  https://huggingface.co/mabote-itumeleng/ONNX-SAM2-Segment-Anything/resolve/main/sam2.1_hiera_large_decoder.onnx \
  -o "$MODEL_DIR/sam2.1_hiera_large_decoder.onnx"

echo "Moving decoder to public directory..."

mv "$MODEL_DIR/sam2.1_hiera_large_decoder.onnx" "$PUBLIC_DIR/"

echo ""
echo "Current directory: $(pwd)"
echo ""

# Add current directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "Starting Uvicorn..."
python -m uvicorn service.main:app --host 0.0.0.0 --port 8188 --reload
