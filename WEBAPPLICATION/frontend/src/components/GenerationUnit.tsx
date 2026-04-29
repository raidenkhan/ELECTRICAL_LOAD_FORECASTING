import React from 'react';

type UnitStatus = 'online' | 'offline' | 'derated';

interface GenerationUnitProps {
  name: string;
  capacity_mw: number;
  dispatch_mw: number;
  status: UnitStatus;
  onToggle?: () => void;
}

export function GenerationUnit({
  name,
  capacity_mw,
  dispatch_mw,
  status,
  onToggle
}: GenerationUnitProps) {
  const statusColors: Record<UnitStatus, string> = {
    online: 'text-[var(--status-emerald)]',
    offline: 'text-[var(--status-crimson)]',
    derated: 'text-[var(--status-amber)]',
  };

  const dispatchPercent = (dispatch_mw / capacity_mw) * 100;

  return (
    <button 
      onClick={onToggle}
      className={`glass-panel p-4 flex flex-col gap-3 min-w-[200px] text-left transition-all duration-300
        ${status === 'offline' ? 'opacity-60 grayscale' : 'hover:border-[var(--brand-indigo)]'}
      `}
    >
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-bold text-[var(--text-primary)]">{name}</span>
        <div className={`w-2 h-2 rounded-full ${statusColors[status].replace('text', 'bg')} ${status === 'online' ? 'animate-pulse' : ''}`} />
      </div>

      <div className="flex flex-col">
        <div className="flex justify-between items-baseline">
          <span className="data-num text-[var(--text-primary)]">{dispatch_mw}</span>
          <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase">/ {capacity_mw} MW</span>
        </div>
        
        {/* Progress Bar */}
        <div className="h-1 w-full bg-[var(--divider)] rounded-full mt-1 overflow-hidden">
          <div 
            className={`h-full transition-all duration-1000 ${statusColors[status].replace('text', 'bg')}`}
            style={{ width: `${dispatchPercent}%` }}
          />
        </div>
      </div>

      <span className={`text-[10px] font-bold uppercase tracking-wider ${statusColors[status]}`}>
        {status}
      </span>
    </button>
  );
}
