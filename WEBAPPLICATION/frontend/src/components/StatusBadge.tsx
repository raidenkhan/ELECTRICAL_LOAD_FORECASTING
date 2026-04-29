import React from 'react';

type RegimeStatus = 'stable' | 'peak' | 'outage' | 'holiday' | 'connecting' | 'warning' | 'critical' | 'emerald';

interface StatusBadgeProps {
  status: RegimeStatus;
  label?: string;
  text?: string; // Support the 'text' prop used in ModelPerformance
  pulse?: boolean;
}

export function StatusBadge({ status, label, text, pulse }: StatusBadgeProps) {
  const configs: Record<RegimeStatus, { color: string; bg: string; text: string }> = {
    stable: {
      color: 'bg-[var(--status-emerald)]',
      bg: 'bg-[var(--status-emerald)]/10',
      text: 'text-[var(--status-emerald)]',
    },
    emerald: {
      color: 'bg-[var(--status-emerald)]',
      bg: 'bg-[var(--status-emerald)]/10',
      text: 'text-[var(--status-emerald)]',
    },
    peak: {
      color: 'bg-[var(--status-amber)]',
      bg: 'bg-[var(--status-amber)]/10',
      text: 'text-[var(--status-amber)]',
    },
    warning: {
      color: 'bg-[var(--status-amber)]',
      bg: 'bg-[var(--status-amber)]/10',
      text: 'text-[var(--status-amber)]',
    },
    critical: {
      color: 'bg-[var(--status-crimson)]',
      bg: 'bg-[var(--status-crimson)]/10',
      text: 'text-[var(--status-crimson)]',
    },
    outage: {
      color: 'bg-[var(--status-crimson)]',
      bg: 'bg-[var(--status-crimson)]/10',
      text: 'text-[var(--status-crimson)]',
    },
    holiday: {
      color: 'bg-[var(--status-violet)]',
      bg: 'bg-[var(--status-violet)]/10',
      text: 'text-[var(--status-violet)]',
    },
    connecting: {
      color: 'bg-[var(--text-muted)]',
      bg: 'bg-[var(--text-muted)]/10',
      text: 'text-[var(--text-muted)]',
    },
  };

  const config = configs[status] || configs.connecting;
  const displayText = text || label || status.toUpperCase();

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-sm border border-[var(--divider)] ${config.bg} ${config.text}`}>
      <div className={`w-1.5 h-1.5 rounded-sm ${config.color} ${pulse ? 'animate-pulse-live shadow-[0_0_8px_currentColor]' : ''}`} />
      <span className="text-[10px] font-bold uppercase tracking-widest font-mono">
        {displayText}
      </span>
    </div>
  );
}
