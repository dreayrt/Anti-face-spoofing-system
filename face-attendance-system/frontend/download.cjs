const https = require('https');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, 'public', 'models');
if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
}

const baseUrl = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model/';
const files = [
    'tiny_face_detector_model-weights_manifest.json',
    'tiny_face_detector_model.weights.bin',
    'face_landmark_68_model-weights_manifest.json',
    'face_landmark_68_model.weights.bin'
];

async function download() {
    for (const file of files) {
        console.log(`Downloading ${file}...`);
        await new Promise((resolve, reject) => {
            const dest = path.join(dir, file);
            const fileStream = fs.createWriteStream(dest);
            https.get(baseUrl + file, (response) => {
                if (response.statusCode !== 200) {
                     reject(new Error(`Failed to download ${file}: ${response.statusCode}`));
                     return;
                }
                response.pipe(fileStream);
                fileStream.on('finish', () => {
                    fileStream.close();
                    resolve();
                });
            }).on('error', (err) => {
                fs.unlink(dest, () => {});
                reject(err);
            });
        });
    }
    console.log("Done downloading models.");
}

download().catch(console.error);
