import { CSVProcessorTool } from './src/bubbles/tool-bubble/csv-processor-tool.js';

async function test() {
  const tool = new CSVProcessorTool({
    operation: 'transform',
    csvData: 'price,quantity\n10,5\n20,3',
    transformRules: [
      {
        column: 'total',
        operation: 'calculate',
        expression: '{price} * {quantity}',
      },
    ],
  });

  const result = await tool.performAction();
  console.log('Success:', result.success);
  console.log('Data:', JSON.stringify(result.data, null, 2));
}

test().catch(console.error);
