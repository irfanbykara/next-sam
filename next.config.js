/** @type {import('next').NextConfig} */

const path = require('path');

const nextConfig = {
    webpack: (config) => {        
        // See https://webpack.js.org/configuration/resolve/#resolvealias
        config.resolve.alias = {
            ...config.resolve.alias,
            "onnxruntime-node$": false,
            '@huggingface/transformers': path.resolve(__dirname, 'node_modules/@huggingface/transformers'),
        }
        return config;
    },
}
module.exports = nextConfig
