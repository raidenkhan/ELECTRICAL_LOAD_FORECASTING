import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: {
    value: number;
    isUp: boolean;
  };
  status?: 'emerald' | 'amber' | 'crimson' | 'none';
  subtext?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  unit,
  trend,
  status = 'none',
  subtext,
  className = ''
}: MetricCardProps) {
  const statusColors = {
    emerald: 'border-l-[var(--status-emerald)]',
    amber: 'border-l-[var(--status-amber)]',
    crimson: 'border-l-[var(--status-crimson)]',
    none: 'border-l-transparent'
  };

  return (
    <div className={`glass-panel p-5 border-l-4 ${statusColors[status]} ${className}`}>
      <div className="flex flex-col gap-1">
        <span className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">
          {label}
        </span>
        
        <div className="flex items-baseline gap-2">
          <span className="metric-num text-[var(--text-primary)]">
            {value}
          </span>
          {unit && (
            <span className="text-[14px] font-bold text-[var(--text-secondary)] uppercase">
              {unit}
            </span>
          )}
          
          {trend && (
            <div className={`flex items-center text-[12px] font-bold ml-auto
              ${trend.isUp ? 'text-[var(--status-emerald)]' : 'text-[var(--status-crimson)]'}
            `}>
              {trend.isUp ? '↑' : '↓'} {Math.abs(trend.value)}%
            </div>
          )}
        </div>

        {subtext && (
          <p className="text-[11px] text-[var(--text-muted)] mt-1 font-medium italic">
            {subtext}
          </p>
        )}
      </div>
    </div>
  );
}
