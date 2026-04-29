import React from 'react';
import { AlertCircle, AlertTriangle, Info } from 'lucide-react';

type AlertSeverity = 'info' | 'warning' | 'critical';

interface AlertBannerProps {
  severity: AlertSeverity;
  message: string;
  className?: string;
}

export function AlertBanner({ severity, message, className = '' }: AlertBannerProps) {
  const configs: Record<AlertSeverity, { bg: string; border: string; text: string; icon: React.ReactNode }> = {
    info: {
      bg: 'bg-[var(--brand-indigo)]/5',
      border: 'border-[var(--brand-indigo)]/20',
      text: 'text-[var(--brand-indigo)]',
      icon: <Info className="w-4 h-4" />
    },
    warning: {
      bg: 'bg-[var(--status-amber)]/10',
      border: 'border-[var(--status-amber)]/30',
      text: 'text-[var(--status-amber)]',
      icon: <AlertTriangle className="w-4 h-4" />
    },
    critical: {
      bg: 'bg-[var(--status-crimson)]/10',
      border: 'border-[var(--status-crimson)]/30',
      text: 'text-[var(--status-crimson)]',
      icon: <AlertCircle className="w-4 h-4 animate-pulse" />
    }
  };

  const config = configs[severity];

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-md border ${config.bg} ${config.border} ${config.text} ${className}`}>
      {config.icon}
      <span className="text-[13px] font-medium">{message}</span>
    </div>
  );
}
