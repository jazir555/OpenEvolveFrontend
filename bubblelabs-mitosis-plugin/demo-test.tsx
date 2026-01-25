import React from 'react';
import ReactDOM from 'react-dom/client';
import { MitosisDemo } from './src/components/MitosisDemo';
import { mitosisPlugin } from './src/utils/createMitosisPlugin';

// Initialize the plugin
mitosisPlugin.initialize({
  enabled: true,
  animationDuration: 1500,
  bounceIntensity: 0.3,
  splitDelay: 300,
  colorVariation: 0.1,
  rotationIntensity: 0.2,
  opacityEffect: true,
  trailEffect: false,
  easingFunction: 'cubic-bezier(0.25, 0.1, 0.25, 1)',
  particleEffects: false
});

// Simple test app
const TestApp = () => {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>OpenEvolve Mitosis Demo</h1>
      <p>This demo shows the "Mitosis" visual functionality:</p>
      <ol>
        <li>A single bubble labeled "Draft Email" splits into 5 bubbles</li>
        <li>4 bubbles turn Red (Failed/Killed)</li>
        <li>1 turns Green (Winner)</li>
        <li>The Green one splits again</li>
      </ol>
      <p><strong>Narrative:</strong> "You are watching the system survival-of-the-fittest its way to the perfect answer. You didn't have to code the retries. The evolution did it for you."</p>
      
      <MitosisDemo 
        enabled={true}
        onDemoComplete={() => console.log('Demo completed!')}
      />
    </div>
  );
};

// Render the app
const container = document.getElementById('root');
if (container) {
  const root = ReactDOM.createRoot(container);
  root.render(<TestApp />);
} else {
  console.error('Could not find root container');
}