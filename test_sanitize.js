const input = "'; DROP TABLE users; --";
const result = input
  .replace(/[<>]/g, '')
  .replace(/[()]/g, '')
  .replace(/['"]/g, '')
  .replace(/;/g, '')
  .replace(/--/g, '')
  .trim();
console.log('Input:', JSON.stringify(input));
console.log('Result:', JSON.stringify(result));
console.log('Expected:', JSON.stringify(' DROP TABLE users '));
