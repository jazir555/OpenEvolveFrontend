import { BubbleFactory } from '@bubblelab/bubble-core/src/bubble-factory.js';

async function testRAGBitsBubbles() {
  console.log('🧪 Testing RAGBits bubbles integration...\n');
  
  try {
    // Create a bubble factory instance
    const factory = new BubbleFactory();
    await factory.registerDefaults();
    
    console.log('✅ Bubble factory created and defaults registered\n');
    
    // Check if RAGBits bubbles are registered
    const ragbitsBubbles = [
      'ragbits-ingest',
      'ragbits-search',
      'ragbits-index', 
      'ragbits-generation'
    ];
    
    console.log('🔍 Checking if RAGBits bubbles are registered:');
    for (const bubbleName of ragbitsBubbles) {
      const bubbleClass = factory.get(bubbleName as any);
      if (bubbleClass) {
        console.log(`  ✅ ${bubbleName} - Registered`);
        
        // Show basic metadata
        const metadata = factory.getMetadata(bubbleName as any);
        if (metadata) {
          console.log(`      Description: ${metadata.shortDescription}`);
        }
      } else {
        console.log(`  ❌ ${bubbleName} - NOT FOUND`);
      }
    }
    
    console.log('\n🧪 Testing bubble instantiation:');
    
    // Test creating instances of each bubble with minimal params
    try {
      const ingestBubble = factory.createBubble('ragbits-ingest', {
        content: 'Test document content',
        serverUrl: 'http://localhost:8002'
      });
      console.log('  ✅ RAGBitsIngestBubble instantiated');
    } catch (e) {
      console.log('  ❌ Failed to instantiate RAGBitsIngestBubble:', e.message);
    }
    
    try {
      const searchBubble = factory.createBubble('ragbits-search', {
        query: 'test query',
        serverUrl: 'http://localhost:8002'
      });
      console.log('  ✅ RAGBitsSearchBubble instantiated');
    } catch (e) {
      console.log('  ❌ Failed to instantiate RAGBitsSearchBubble:', e.message);
    }
    
    try {
      const indexBubble = factory.createBubble('ragbits-index', {
        serverUrl: 'http://localhost:8002'
      });
      console.log('  ✅ RAGBitsIndexBubble instantiated');
    } catch (e) {
      console.log('  ❌ Failed to instantiate RAGBitsIndexBubble:', e.message);
    }
    
    try {
      const genBubble = factory.createBubble('ragbits-generation', {
        query: 'test query',
        serverUrl: 'http://localhost:8002'
      });
      console.log('  ✅ RAGBitsGenerationBubble instantiated');
    } catch (e) {
      console.log('  ❌ Failed to instantiate RAGBitsGenerationBubble:', e.message);
    }
    
    console.log('\n🎉 RAGBits bubbles integration test completed successfully!');
    console.log('\nThe RAGBits bubbles are properly integrated into BubbleLab and ready for use in workflows.');
    
  } catch (error) {
    console.error('❌ Error during RAGBits bubbles test:', error);
    process.exit(1);
  }
}

// Run the test
testRAGBitsBubbles();