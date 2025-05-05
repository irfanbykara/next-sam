export function maskImageCanvas(imageCanvas, maskCanvas) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.height = imageCanvas.height;
  canvas.width = imageCanvas.width;

  context.drawImage(
    maskCanvas,
    0,
    0,
    maskCanvas.width,
    maskCanvas.height,
    0,
    0,
    canvas.width,
    canvas.height
  );
  context.globalCompositeOperation = "source-in";
  context.drawImage(
    imageCanvas,
    0,
    0,
    imageCanvas.width,
    imageCanvas.height,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return canvas;
}

export function resizeCanvas(canvasOrig, size) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = size.h;
  canvas.width = size.w;

  ctx.drawImage(
    canvasOrig,
    0,
    0,
    canvasOrig.width,
    canvasOrig.height,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return canvas;
}

// input: 2x Canvas, output: One new Canvas, resize source
export function mergeMasks(sourceMask, targetMask) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = targetMask.height;
  canvas.width = targetMask.width;

  ctx.drawImage(targetMask, 0, 0);
  ctx.drawImage(
    sourceMask,
    0,
    0,
    sourceMask.width,
    sourceMask.height,
    0,
    0,
    targetMask.width,
    targetMask.height
  );

  return canvas;
}

// input: source and target {w, h}, output: {x,y,w,h} to fit source nicely into target preserving aspect
export function resizeAndPadBox(sourceDim, targetDim) {
  if (sourceDim.h == sourceDim.w) {
    return { x: 0, y: 0, w: targetDim.w, h: targetDim.h };
  } else if (sourceDim.h > sourceDim.w) {
    // portrait => resize and pad left
    const newW = (sourceDim.w / sourceDim.h) * targetDim.w;
    const padLeft = Math.floor((targetDim.w - newW) / 2);

    return { x: padLeft, y: 0, w: newW, h: targetDim.h };
  } else if (sourceDim.h < sourceDim.w) {
    // landscape => resize and pad top
    const newH = (sourceDim.h / sourceDim.w) * targetDim.h;
    const padTop = Math.floor((targetDim.h - newH) / 2);

    return { x: 0, y: padTop, w: targetDim.w, h: newH };
  }
}

/** 
 * input: onnx Tensor [B, *, W, H] and index idx
 * output: Tensor [B, idx, W, H]
 **/
export function sliceTensor(tensor, idx) {
  const [bs, noMasks, width, height] = tensor.dims;
  const stride = width * height;
  const start = stride * idx,
    end = start + stride;

  return tensor.cpuData.slice(start, end);
}

// // Apply Gaussian blur
// export function applyGaussianBlur(imageData, width, height, radius = 3) {
//   const canvas = document.createElement("canvas");
//   const ctx = canvas.getContext("2d");

//   canvas.width = width;
//   canvas.height = height;

//   // Draw the original mask
//   ctx.putImageData(imageData, 0, 0);

//   // Apply Gaussian blur
//   ctx.filter = `blur(${radius}px)`;
//   ctx.drawImage(canvas, 0, 0);

//   // Get the blurred image data
//   return ctx.getImageData(0, 0, width, height);
// }

// Process mask canvas: apply blur
export function processMaskCanvas(maskCanvas) {
  const ctx = maskCanvas.getContext("2d");
  const width = maskCanvas.width;
  const height = maskCanvas.height;

  // Get mask image data
  let maskData = ctx.getImageData(0, 0, width, height);

  // Apply Gaussian blur
  let blurredMask = applyGaussianBlur(maskData, width, height, 1);

  ctx.putImageData(blurredMask, 0, 0);

  return maskCanvas;
}


/** 
 * input: HTMLCanvasElement (RGB)
 * output: Float32Array for later conversion to ORT.Tensor of shape [1, 3, canvas.width, canvas.height]
 *  
 * inspired by: https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
 **/ 
export function canvasToFloat32Array(canvas) {
  const imageData = canvas
    .getContext("2d")
    .getImageData(0, 0, canvas.width, canvas.height).data;
  const shape = [1, 3, canvas.width, canvas.height];

  const [redArray, greenArray, blueArray] = [[], [], []];

  for (let i = 0; i < imageData.length; i += 4) {
    redArray.push(imageData[i]);
    greenArray.push(imageData[i + 1]);
    blueArray.push(imageData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  const transposedData = redArray.concat(greenArray).concat(blueArray);

  let i,
    l = transposedData.length;
  const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);
  for (i = 0; i < l; i++) {
    float32Array[i] = transposedData[i] / 255.0; // convert to float
  }

  return { float32Array, shape };
}

export function canvasToBase64(canvas, format = "image/png", quality = 1.0) {
   console.log("Inside the base64 converter")
  const dataURL = canvas.toDataURL(format, quality);

  return dataURL.split(',')[1];  // remove the "data:image/png;base64," prefix
  // Print the first 10 characters of the base64 string to the console
  
}


/** 
 * input: HTMLCanvasElement (RGB)
 * output: Float32Array for later conversion to ORT.Tensor of shape [1, 3, canvas.width, canvas.height]
 *  
 * inspired by: https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
 **/ 
export function maskCanvasToFloat32Array(canvas) {
  const imageData = canvas
    .getContext("2d")
    .getImageData(0, 0, canvas.width, canvas.height).data;

  const shape = [1, 1, canvas.width, canvas.height];
  const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);

  for (let i = 0; i < float32Array.length; i++) {
    float32Array[i] = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / (3 * 255.0); // convert avg to float
  }

  return float32Array;
}




export function base64ToFloat32Array(base64Str) {
    let binaryString = atob(base64Str); // Decode Base64 to binary string
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer); // Convert to Float32Array
}



export function float32ArrayToCanvas(array, width, height) {
  const C = 4; // 4 output channels, RGBA
  const imageData = new Uint8ClampedArray(array.length * C);

  // Threshold to identify masked pixels
  for (let srcIdx = 0; srcIdx < array.length; srcIdx++) {
    const trgIdx = srcIdx * C;
    const maskedPx = array[srcIdx] > 0;
    imageData[trgIdx] = maskedPx ? 0x32 : 0;  // Green Channel
    imageData[trgIdx + 1] = maskedPx ? 0xcd : 0;  // Red Channel
    imageData[trgIdx + 2] = maskedPx ? 0x32 : 0;  // Blue Channel
    imageData[trgIdx + 3] = maskedPx ? 255 : 0;  // Alpha Channel
  }

  // Create canvas and apply image data
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = height;
  canvas.width = width;
  ctx.putImageData(new ImageData(imageData, width, height), 0, 0);

  // Step 1: Apply one erosion
  applyErosion(ctx, width, height);

  // Step 2: Apply Gaussian blur
  applyGaussianBlur(ctx, width, height);

  return canvas;
}

function applyErosion(ctx, width, height) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const pixels = imageData.data;
  const tempPixels = new Uint8ClampedArray(pixels);

  for (let i = 0; i < pixels.length; i += 4) {
    const x = (i / 4) % width;
    const y = Math.floor(i / 4 / width);

    // If the pixel is part of the mask (non-zero alpha value), check if its neighbors are also part of the mask
    if (pixels[i + 3] === 255) {
      const neighbors = [
        { x: x - 1, y: y }, // left
        { x: x + 1, y: y }, // right
        { x: x, y: y - 1 }, // top
        { x: x, y: y + 1 }, // bottom
      ];

      let hasNeighbor = false;
      for (let neighbor of neighbors) {
        if (neighbor.x >= 0 && neighbor.x < width && neighbor.y >= 0 && neighbor.y < height) {
          const neighborIndex = (neighbor.y * width + neighbor.x) * 4;
          if (pixels[neighborIndex + 3] === 255) {
            hasNeighbor = true;
            break;
          }
        }
      }

      // If no neighbor is part of the mask, remove this pixel (set it to transparent)
      if (!hasNeighbor) {
        tempPixels[i + 1] = 0;
        tempPixels[i + 2] = 0;
        tempPixels[i + 3] = 0;
      }
    }
  }

  // Update the canvas with the modified pixels
  ctx.putImageData(new ImageData(tempPixels, width, height), 0, 0);
}


function applyGaussianBlur(ctx, width, height) {
  // Apply a Gaussian blur using canvas' filter property
  // The filter property applies CSS filters to the canvas, like blur
  ctx.filter = "blur(2px)"; // Adjust the blur radius as needed
  ctx.drawImage(ctx.canvas, 0, 0);
  ctx.filter = "none";  // Reset filter to prevent applying further filters
}
