/**
 * Notification Component
 * Toast-style notifications for user feedback
 */

import { useEffect, useState } from 'react';

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  duration?: number;
}

interface NotificationsProps {
  notifications: Notification[];
  onClose: (id: string) => void;
}

export function Notifications({ notifications, onClose }: NotificationsProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <NotificationCard
          key={notification.id}
          notification={notification}
          onClose={() => onClose(notification.id)}
        />
      ))}
    </div>
  );
}

function NotificationCard({
  notification,
  onClose,
}: {
  notification: Notification;
  onClose: () => void;
}) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const duration = notification.duration || 5000;
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300);
    }, duration);

    return () => clearTimeout(timer);
  }, [notification.duration, onClose]);

  const getStyles = () => {
    switch (notification.type) {
      case 'success':
        return 'border-green-500 bg-green-50 dark:bg-green-900/20';
      case 'error':
        return 'border-red-500 bg-red-50 dark:bg-red-900/20';
      case 'warning':
        return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
      default:
        return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20';
    }
  };

  const getIcon = () => {
    switch (notification.type) {
      case 'success':
        return (
          <svg className="h-5 w-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-5 w-5 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        );
      case 'warning':
        return (
          <svg className="h-5 w-5 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        );
      default:
        return (
          <svg className="h-5 w-5 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        );
    }
  };

  return (
    <div
      className={`
        flex w-96 max-w-sm items-start gap-3 rounded-lg border p-4 shadow-lg
        transition-all duration-300
        ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'}
        ${getStyles()}
      `}
    >
      <div className="flex-shrink-0">{getIcon()}</div>

      <div className="flex-1">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
          {notification.title}
        </h4>
        <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">
          {notification.message}
        </p>
      </div>

      <button
        onClick={() => {
          setIsVisible(false);
          setTimeout(onClose, 300);
        }}
        className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      >
        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </div>
  );
}

// Global notification manager
let notificationListeners: Set<(notifications: Notification[]) => void> = new Set();
let notificationList: Notification[] = [];

export function notify(notification: Omit<Notification, 'id'>) {
  const id = Math.random().toString(36).substring(7);
  const newNotification = { ...notification, id };
  notificationList = [...notificationList, newNotification];
  notificationListeners.forEach((listener) => listener(notificationList));
}

export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  useEffect(() => {
    notificationListeners.add(setNotifications);
    setNotifications(notificationList);

    return () => {
      notificationListeners.delete(setNotifications);
    };
  }, []);

  const handleClose = (id: string) => {
    notificationList = notificationList.filter((n) => n.id !== id);
    notificationListeners.forEach((listener) => listener(notificationList));
  };

  return { notifications, handleClose };
}
