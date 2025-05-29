setup:
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
	. $$HOME/.nvm/nvm.sh && \
	nvm install node && \
	npm install next
	curl -L -H "Authorization: Bearer $$HF_TOKEN" \
    https://huggingface.co/mabote-itumeleng/ONNX-SAM2-Segment-Anything/resolve/main/sam2.1_hiera_large_decoder.onnx \
    -o sam2.1_hiera_large_decoder.onnx
	mv sam2.1_hiera_large_decoder.onnx public

build:
	. $$HOME/.nvm/nvm.sh && \
	npm run build

start:
	. $$HOME/.nvm/nvm.sh && \
	npm start
