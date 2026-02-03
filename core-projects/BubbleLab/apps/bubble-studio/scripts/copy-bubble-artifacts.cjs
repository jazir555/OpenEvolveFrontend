const fs = require('fs');
const path = require('path');

const studioRoot = path.resolve(__dirname, '..');
const sourceRoot = path.resolve(
  studioRoot,
  '..',
  '..',
  'packages',
  'bubble-core',
  'dist'
);
const publicDir = path.resolve(studioRoot, 'public');

const files = [
  {
    source: path.join(sourceRoot, 'bubble-bundle.d.ts'),
    destination: path.join(publicDir, 'bubble-types.txt'),
  },
  {
    source: path.join(sourceRoot, 'bubbles.json'),
    destination: path.join(publicDir, 'bubbles.json'),
  },
];

fs.mkdirSync(publicDir, { recursive: true });

files.forEach(({ source, destination }) => {
  if (!fs.existsSync(source)) {
    throw new Error(`Missing artifact: ${source}`);
  }
  fs.copyFileSync(source, destination);
});

console.log('Copied BubbleLab artifacts into bubble-studio/public.');
