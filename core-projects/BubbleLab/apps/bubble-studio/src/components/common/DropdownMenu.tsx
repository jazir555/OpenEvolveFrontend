/**
 * DropdownMenu Component
 * Context menu dropdown
 */

import { useState, useRef, useEffect } from 'react';

interface DropdownMenuProps {
  trigger: React.ReactNode;
  items: DropdownMenuItem[];
  align?: 'left' | 'right';
}

interface DropdownMenuItem {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  divider?: boolean;
}

export function DropdownMenu({ trigger, items, align = 'left' }: DropdownMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const alignStyles = {
    left: 'left-0',
    right: 'right-0',
  };

  return (
    <div className="relative inline-block text-left">
      <div onClick={() => setIsOpen(!isOpen)}>
        {trigger}
      </div>

      {isOpen && (
        <div
          ref={menuRef}
          className={`absolute ${alignStyles[align]} z-50 mt-2 w-56 origin-top rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none dark:bg-gray-800`}
        >
          <div className="py-1">
            {items.map((item, index) => {
              if (item.divider) {
                return (
                  <div key={index} className="border-t border-gray-100 dark:border-gray-700 my-1" />
                );
              }

              return (
                <button
                  key={index}
                  onClick={() => {
                    item.onClick();
                    setIsOpen(false);
                  }}
                  disabled={item.disabled}
                  className={`
                    group flex w-full items-center px-4 py-2 text-sm text-gray-700
                    hover:bg-gray-100 hover:text-gray-900
                    dark:text-gray-300 dark:hover:bg-gray-700
                    disabled:opacity-50 disabled:cursor-not-allowed
                  `}
                >
                  {item.icon && (
                    <span className="mr-2 text-gray-400 group-hover:text-gray-500 dark:text-gray-500">
                      {item.icon}
                    </span>
                  )}
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
