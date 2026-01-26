/**
 * Accordion Component
 * Collapsible content sections
 */

import { useState } from 'react';

interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  disabled?: boolean;
}

interface AccordionProps {
  items: AccordionItem[];
  multiple?: boolean;
  defaultOpen?: string[];
  className?: string;
}

export function Accordion({
  items,
  multiple = false,
  defaultOpen = [],
  className = '',
}: AccordionProps) {
  const [openItems, setOpenItems] = useState<Set<string>>(new Set(defaultOpen));

  const toggleItem = (id: string) => {
    if (multiple) {
      const newOpen = new Set(openItems);
      if (newOpen.has(id)) {
        newOpen.delete(id);
      } else {
        newOpen.add(id);
      }
      setOpenItems(newOpen);
    } else {
      setOpenItems(new Set(openItems.has(id) ? [] : [id]));
    }
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {items.map((item) => (
        <div
          key={item.id}
          className={`border border-gray-300 dark:border-gray-700 rounded-lg overflow-hidden ${
            item.disabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <button
            onClick={() => !item.disabled && toggleItem(item.id)}
            disabled={item.disabled}
            className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-expanded={openItems.has(item.id)}
          >
            <span className="font-medium text-gray-900 dark:text-white">
              {item.title}
            </span>
            <svg
              className={`w-5 h-5 text-gray-500 transition-transform ${
                openItems.has(item.id) ? 'transform rotate-180' : ''
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>
          {openItems.has(item.id) && (
            <div className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-300 dark:border-gray-700">
              {item.content}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
