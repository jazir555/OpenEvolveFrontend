/**
 * Drawer Component
 * Slide-out panel from edge of screen
 */

import { useEffect, useRef } from 'react';
import { useClickOutside } from '../../hooks/useClickOutside';

interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  placement?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: React.ReactNode;
}

export function Drawer({
  isOpen,
  onClose,
  placement = 'right',
  size = 'md',
  children,
}: DrawerProps) {
  const drawerRef = useClickOutside<HTMLDivElement>(onClose, isOpen);

  const sizeStyles = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-full',
  };

  const placementStyles = {
    left: 'left-0 top-0 bottom-0 border-r',
    right: 'right-0 top-0 bottom-0 border-l',
    top: 'top-0 left-0 right-0 border-b',
    bottom: 'bottom-0 left-0 right-0 border-t',
  };

  const transformStyles = {
    left: isOpen ? 'translate-x-0' : '-translate-x-full',
    right: isOpen ? 'translate-x-0' : 'translate-x-full',
    top: isOpen ? 'translate-y-0' : '-translate-y-full',
    bottom: isOpen ? 'translate-y-0' : 'translate-y-full',
  };

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        ref={drawerRef}
        className={`
          fixed ${placementStyles[placement]} ${sizeStyles[size]}
          bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700
          shadow-2xl z-50
          transition-transform duration-300 ease-in-out
          ${transformStyles[placement]}
          ${placement === 'left' || placement === 'right' ? 'h-full' : 'w-full'}
        `}
      >
        {children}
      </div>
    </>
  );
}

interface DrawerHeaderProps {
  title: string;
  onClose: () => void;
}

export function DrawerHeader({ title, onClose }: DrawerHeaderProps) {
  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
        {title}
      </h2>
      <button
        onClick={onClose}
        className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </div>
  );
}

interface DrawerBodyProps {
  children: React.ReactNode;
  className?: string;
}

export function DrawerBody({ children, className = '' }: DrawerBodyProps) {
  return (
    <div className={`flex-1 overflow-auto p-4 ${className}`}>
      {children}
    </div>
  );
}

interface DrawerFooterProps {
  children: React.ReactNode;
  className?: string;
}

export function DrawerFooter({ children, className = '' }: DrawerFooterProps) {
  return (
    <div className={`px-4 py-3 border-t border-gray-200 dark:border-gray-700 ${className}`}>
      {children}
    </div>
  );
}
