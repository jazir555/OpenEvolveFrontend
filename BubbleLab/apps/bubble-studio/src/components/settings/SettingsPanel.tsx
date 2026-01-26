/**
 * Settings Panel Component
 * Application and LLM configuration settings
 */

import { useState } from 'react';
import { useLLMConfig } from '../../stores/configStore';
import { LLMProvider } from '../../types/api';

export function SettingsPanel() {
  const {
    provider,
    api_key,
    base_url,
    model_leanaide,
    model_text,
    model_img,
    temperature,
    top_p,
    max_tokens,
    frequency_penalty,
    presence_penalty,
    setLLMProvider,
    setApiKey,
    setBaseUrl,
    setModelLeanAide,
    setModelText,
    setModelImg,
    setTemperature,
    setTopP,
    setMaxTokens,
    setFrequencyPenalty,
    setPresencePenalty,
  } = useLLMConfig();

  const [showApiKey, setShowApiKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsSaving(false);
  };

  return (
    <div className="space-y-6">
      {/* LLM Provider Settings */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          LLM Configuration
        </h2>

        <div className="space-y-4">
          {/* Provider Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Provider
            </label>
            <select
              value={provider}
              onChange={(e) => setLLMProvider(e.target.value as LLMProvider)}
              className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
              <option value={LLMProvider.OPENAI}>OpenAI</option>
              <option value={LLMProvider.ANTHROPIC}>Anthropic</option>
              <option value={LLMProvider.COHERE}>Cohere</option>
              <option value={LLMProvider.CUSTOM}>Custom</option>
            </select>
          </div>

          {/* API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              API Key
            </label>
            <div className="mt-1 flex rounded-md shadow-sm">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={api_key}
                onChange={(e) => setApiKey(e.target.value)}
                className="block w-full rounded-l-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="inline-flex items-center rounded-r-lg border border-l-0 border-gray-300 bg-gray-50 px-3 text-sm text-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-600 dark:text-gray-300 dark:hover:bg-gray-500"
              >
                {showApiKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          {/* Base URL (optional) */}
          {provider === LLMProvider.CUSTOM && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Base URL
              </label>
              <input
                type="text"
                value={base_url || ''}
                onChange={(e) => setBaseUrl(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="https://api.example.com"
              />
            </div>
          )}

          {/* Models */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                LeanAide Model
              </label>
              <input
                type="text"
                value={model_leanaide}
                onChange={(e) => setModelLeanAide(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-4"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Text Model
              </label>
              <input
                type="text"
                value={model_text}
                onChange={(e) => setModelText(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-3.5-turbo"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Image Model
              </label>
              <input
                type="text"
                value={model_img}
                onChange={(e) => setModelImg(e.target.value)}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                placeholder="gpt-4-vision-preview"
              />
            </div>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Top P: {top_p}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={top_p}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Max Tokens
              </label>
              <input
                type="number"
                min="1"
                value={max_tokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Frequency Penalty: {frequency_penalty}
              </label>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={frequency_penalty}
                onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                className="mt-1 block w-full"
              />
            </div>
          </div>

          {/* Save Button */}
          <div className="pt-4">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSaving ? 'Saving...' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>

      {/* UI Preferences */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          UI Preferences
        </h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Auto-save
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Automatically save your work
              </p>
            </div>
            <button
              onClick={() => {
                // Toggle auto-save in config store
              }}
              className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-blue-600"
              role="switch"
              aria-checked="true"
            >
              <span
                aria-hidden="true"
                className="translate-x-5 pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Dark Mode
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Use dark theme
              </p>
            </div>
            <button
              onClick={() => {
                // Toggle dark mode
              }}
              className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 bg-gray-200"
              role="switch"
              aria-checked="false"
            >
              <span
                aria-hidden="true"
                className="translate-x-0 pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
