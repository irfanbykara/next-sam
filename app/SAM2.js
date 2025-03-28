import path from "path";
import pako from "pako";
import * as ort from "onnxruntime-web/all";
import { base64ToFloat32Array} from "@/lib/imageutils";

const DECODER_URL = "/sam2.1_hiera_large_decoder.onnx";

export class SAM2 {
  bufferDecoder = null;
  sessionDecoder = null;
  image_encoded = null;

  constructor() {}

  async downloadModels() {
    this.bufferDecoder = await this.downloadModel(DECODER_URL);
  }

  async downloadModel(url) {
    // step 1: check if cached
    const root = await navigator.storage.getDirectory();
    const filename = path.basename(url);

    let fileHandle = await root
      .getFileHandle(filename)
      .catch((e) => console.error("File does not exist:", filename, e));

    if (fileHandle) {
      const file = await fileHandle.getFile();
      if (file.size > 0) return await file.arrayBuffer();
    }

    // step 2: download if not cached
    console.log("File not in cache, downloading from " + url);
    let buffer = null;
    try {
      buffer = await fetch(url, {
        headers: new Headers({
          Origin: location.origin,
        }),
        mode: "cors",
      }).then((response) => response.arrayBuffer());
    } catch (e) {
      console.error("Download of " + url + " failed: ", e);
      return null;
    }

    // step 3: store
    try {
      const fileHandle = await root.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(buffer);
      await writable.close();

      console.log("Stored " + filename);
    } catch (e) {
      console.error("Storage of " + filename + " failed: ", e);
    }

    return buffer;
  }

    async createSessions() {
      const success = await this.getDecoderSession();
    
      return {
        success: success,
        device: success ? this.sessionDecoder[1] : null, // Assuming sessionDecoder[1] stores device info
      };
    }

  async getORTSession(model) {
    /** Creating a session with executionProviders: {"webgpu", "cpu"} fails
     *  => "Error: multiple calls to 'initWasm()' detected."
     *  but ONLY in Safari and Firefox (wtf)
     *  seems to be related to web worker, see https://github.com/microsoft/onnxruntime/issues/22113
     *  => loop through each ep, catch e if not available and move on
     */
    let session = null;
    for (let ep of ["webgpu", "cpu"]) {
      try {
        session = await ort.InferenceSession.create(model, {
          executionProviders: [ep],
        });
      } catch (e) {
        console.error(e);
        continue;
      }

      return [session, ep];
    }
  }


  async getDecoderSession() {
    if (!this.sessionDecoder)
      this.sessionDecoder = await this.getORTSession(this.bufferDecoder);

    return this.sessionDecoder;
  }
    

    async encodeImage(base64Data) {
      try {
              
        // Prepare payload
        const payload = {
          base64: base64Data,
        };
    
        // Convert payload to JSON string
        const jsonString = JSON.stringify(payload);
    
        // Send the Base64-encoded tensor as JSON
        const response = await fetch("https://4d9hagp2o1plq4-8188.proxy.runpod.net/encode/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: jsonString,
        });

        if (!response.ok) throw new Error("Failed to encode image");
            // Check the size of the response before parsing it

      let data = await response.json();

        function base64ToFloat32Array(base64Str) {
            let binaryString = atob(base64Str); // Decode Base64 to binary string
            let len = binaryString.length;
            let bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return new Float32Array(bytes.buffer); // Convert to Float32Array
        }

        let high_res_feats_0 = base64ToFloat32Array(data.high_res_feats_0);
        let high_res_feats_1 = base64ToFloat32Array(data.high_res_feats_1);
        let image_embed = base64ToFloat32Array(data.image_embed);
        
        // Convert to ONNX tensors
        this.image_encoded = {
          high_res_feats_0: new ort.Tensor("float32", high_res_feats_0, [1, 32, 256, 256]),
          high_res_feats_1: new ort.Tensor("float32", high_res_feats_1, [1, 64, 128, 128]),
          image_embed: new ort.Tensor("float32", image_embed, [1, 256, 64, 64]),
        };
      } catch (e) {
        console.error("Encoding error:", e);
      }
    }
    
  async decode(points, masks) {
    const [session, device] = await this.getDecoderSession();

    const flatPoints = points.map((point) => {
      return [point.x, point.y];
    });

    const flatLabels = points.map((point) => {
      return point.label;
    });

    console.log({
      flatPoints,
      flatLabels,
      masks
    });

    let mask_input, has_mask_input
    if (masks) {
      mask_input = masks
      has_mask_input = new ort.Tensor("float32", [1], [1])
    } else {
      // dummy data
      mask_input = new ort.Tensor(
        "float32",
        new Float32Array(256 * 256),
        [1, 1, 256, 256]
      )
      has_mask_input = new ort.Tensor("float32", [0], [1])
    }

    const inputs = {
      image_embed: this.image_encoded.image_embed,
      high_res_feats_0: this.image_encoded.high_res_feats_0,
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new ort.Tensor("float32", flatPoints.flat(), [
        1,
        flatPoints.length,
        2,
      ]),
      point_labels: new ort.Tensor("float32", flatLabels, [
        1,
        flatLabels.length,
      ]),
      mask_input: mask_input,
      has_mask_input: has_mask_input,
    };

    return await session.run(inputs);
  }
}







