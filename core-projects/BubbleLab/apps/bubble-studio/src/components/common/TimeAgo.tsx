/**
 * TimeAgo Component
 * Displays relative time with auto-update
 */

import { useEffect, useState } from 'react';

interface TimeAgoProps {
  date: string | Date;
  updateInterval?: number;
  className?: string;
}

export function TimeAgo({
  date,
  updateInterval = 60000,
  className = '',
}: TimeAgoProps) {
  const [timeAgo, setTimeAgo] = useState(() => formatTimeAgo(date));

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeAgo(formatTimeAgo(date));
    }, updateInterval);

    return () => clearInterval(interval);
  }, [date, updateInterval]);

  return (
    <span className={className} title={new Date(date).toLocaleString()}>
      {timeAgo}
    </span>
  );
}

function formatTimeAgo(date: string | Date): string {
  const now = new Date();
  const past = new Date(date);
  const diffMs = now.getTime() - past.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffMonths = Math.floor(diffDays / 30);
  const diffYears = Math.floor(diffDays / 365);

  if (diffSecs < 60) {
    return 'just now';
  } else if (diffMins < 60) {
    return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  } else if (diffDays < 30) {
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  } else if (diffMonths < 12) {
    return `${diffMonths} month${diffMonths !== 1 ? 's' : ''} ago`;
  } else {
    return `${diffYears} year${diffYears !== 1 ? 's' : ''} ago`;
  }
}
