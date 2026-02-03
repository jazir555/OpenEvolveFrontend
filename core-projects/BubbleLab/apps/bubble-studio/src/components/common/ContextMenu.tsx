/**
 * ContextMenu Component
 * Right-click context menu
 */

import { useState, useEffect, useRef } from 'react';

interface ContextMenuItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
  action: () => void;
}

interface ContextMenuProps {
  items: ContextMenuItem[];
  children: React.ReactNode;
}

export function ContextMenu({ items, children }: ContextMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const menuRef = useRef<HTMLDivElement>(null);

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setPosition({ x: e.clientX, y: e.clientY });
    setIsOpen(true);
  };

  const handleClick = () => {
    setIsOpen(false);
  };

  const handleItemClick = (item: ContextMenuItem) => {
    if (!item.disabled) {
      item.action();
      setIsOpen(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      const handleClickOutside = (e: MouseEvent) => {
        if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
          setIsOpen(false);
        }
      };

      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  return (
    <div onContextMenu={handleContextMenu} onClick={handleClick}>
      {children}

      {isOpen && (
        <div
          ref={menuRef}
          className="fixed z-50 min-w-[180px] bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-1"
          style={{
            left: `${position.x}px`,
            top: `${position.y}px`,
          }}
        >
          {items.map((item) => (
            <button
              key={item.id}
              onClick={() => handleItemClick(item)}
              disabled={item.disabled}
              className={`w-full flex items-center px-3 py-2 text-sm transition-colors ${
                item.disabled
                  ? 'text-gray-400 cursor-not-allowed'
                  : item.danger
                  ? 'text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20'
                  : 'text-gray-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {item.icon && <span className="flex-shrink-0 mr-2">{item.icon}</span>}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
