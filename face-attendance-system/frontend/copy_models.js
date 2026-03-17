const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, 'node_modules', '@vladmandic', 'face-api', 'model');
const publicDir = path.join(__dirname, 'public');
const destDir = path.join(publicDir, 'models');

if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir);
}
if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir);
}

const files = fs.readdirSync(srcDir);
for (const file of files) {
    if (file.startsWith('tiny_face_detector_model') || file.startsWith('face_landmark_68_model')) {
        const srcPath = path.join(srcDir, file);
        const destPath = path.join(destDir, file);
        fs.copyFileSync(srcPath, destPath);
        console.log(`Copied ${file}`);
    }
}
console.log('Copy complete!');
