/**
 * BubbleLab Compatible Tabs Component
 * 
 * A React tabs component that follows BubbleLab's design system
 */

import React, { useState, createContext, useContext } from 'react';

interface TabsContextProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextProps | undefined>(undefined);

interface BubbleTabsProps {
  children: React.ReactNode;
  value: string;
  onValueChange: (value: string) => void;
}

const BubbleTabs: React.FC<BubbleTabsProps> = ({ children, value, onValueChange }) => {
  return (
    <TabsContext.Provider value={{ activeTab: value, setActiveTab: onValueChange }}>
      <div className="w-full">
        {React.Children.map(children, child => {
          if (React.isValidElement(child)) {
            return React.cloneElement(child, { activeTab: value, setActiveTab: onValueChange });
          }
          return child;
        })}
      </div>
    </TabsContext.Provider>
  );
};

interface BubbleTabProps {
  value: string;
  label: string;
  children: React.ReactNode;
  activeTab?: string;
  setActiveTab?: (value: string) => void;
}

const BubbleTab: React.FC<BubbleTabProps> = ({ value, label, children, activeTab, setActiveTab }) => {
  const context = useContext(TabsContext);
  const isActive = (context ? context.activeTab : activeTab) === value;

  const handleClick = () => {
    if (context) {
      context.setActiveTab(value);
    } else if (setActiveTab) {
      setActiveTab(value);
    }
  };

  return (
    <div className="w-full">
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            onClick={handleClick}
            className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
              isActive
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300 dark:hover:border-gray-600'
            }`}
          >
            {label}
          </button>
        </nav>
      </div>
      {isActive && <div className="mt-4">{children}</div>}
    </div>
  );
};

export { BubbleTabs, BubbleTab };