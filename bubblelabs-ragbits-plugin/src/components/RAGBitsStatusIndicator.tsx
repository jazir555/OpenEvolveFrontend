// RAGBits Status Indicator Component

import React from 'react';
import { CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';
import type { RAGBitsStatusIndicatorProps } from '../types/plugin-types';

interface StatusIndicatorProps extends RAGBitsStatusIndicatorProps {
  status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';
}

export const RAGBitsStatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  className = '',
  showDetails = false
}) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'idle':
        return {
          icon: Clock,
          color: 'gray',
          label: 'Idle'
        };
      case 'initializing':
        return {
          icon: Clock,
          color: 'blue',
          label: 'Initializing'
        };
      case 'ready':
        return {
          icon: CheckCircle,
          color: 'green',
          label: 'Ready'
        };
      case 'error':
        return {
          icon: XCircle,
          color: 'red',
          label: 'Error'
        };
      case 'busy':
        return {
          icon: AlertCircle,
          color: 'yellow',
          label: 'Busy'
        };
      default:
        return {
          icon: Clock,
          color: 'gray',
          label: 'Unknown'
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div className={`ragbits-status-indicator ${className}`} data-status={status}>
      <Icon className={`status-icon status-${config.color}`} />
      {showDetails && (
        <span className="status-label">{config.label}</span>
      )}
    </div>
  );
};
