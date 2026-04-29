import { Bell, User, ChevronRight, Moon, Sun, Activity } from 'lucide-react';
import { useSystem } from '@/context/SystemContext';

interface TopBarProps {
  breadcrumbs: string[];
  userName?: string;
  onLogout?: () => void;
  theme?: string;
  onThemeToggle?: () => void;
  viewMode?: 'analytics' | 'control-room';
  onViewModeChange?: (mode: 'analytics' | 'control-room') => void;
}

export function TopBar({ 
  breadcrumbs, 
  onLogout, 
  theme, 
  onThemeToggle, 
  viewMode, 
  onViewModeChange 
}: TopBarProps) {
  const { lastSync, secondsSinceSync, isStale } = useSystem();

  const formatTime = (date: Date | null) => {
    if (!date) return '--:--:--';
    return date.toLocaleTimeString([], { hour12: false });
  };

  const getSyncColor = () => {
    if (isStale) return 'text-[var(--status-crimson)]';
    if (secondsSinceSync > 300) return 'text-[var(--status-amber)]'; // 5 mins
    return 'text-[var(--status-emerald)]';
  };

  return (
    <div className="h-[52px] px-6 flex items-center justify-between border-b bg-white/40 dark:bg-gray-900/40 backdrop-blur-2xl border-white/10 dark:border-white/5 z-40">

      {/* Breadcrumb Cluster */}
      <div className="flex items-center gap-2">
        {breadcrumbs.map((crumb, index) => (
          <div key={crumb} className="flex items-center gap-2">
            {index > 0 && <ChevronRight className="w-3 h-3 text-[var(--text-muted)]" />}
            <span className={`text-[12px] font-medium tracking-tight
              ${index === breadcrumbs.length - 1 ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)]'}
            `}>
              {crumb}
            </span>
          </div>
        ))}
      </div>

      {/* Center Toggle (Themed Tactical Switch) */}
      <div className="hidden lg:flex items-center bg-[var(--surface-secondary)]/60 p-1 rounded-sm border border-[var(--divider)]">
        <button
          onClick={() => onViewModeChange?.('analytics')}
          className={`px-4 py-1.5 rounded-sm text-[11px] font-black uppercase tracking-[0.2em] transition-all duration-200
            ${viewMode === 'analytics' 
              ? 'bg-[var(--brand-blue)] text-white shadow-[0_0_15px_rgba(0,51,153,0.3)]' 
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
        >
          Analytics
        </button>
        <button
          onClick={() => onViewModeChange?.('control-room')}
          className={`px-4 py-1.5 rounded-sm text-[11px] font-black uppercase tracking-[0.2em] transition-all duration-200
            ${viewMode === 'control-room' 
              ? 'bg-[var(--brand-blue)] text-white shadow-[0_0_15px_rgba(0,51,153,0.3)]' 
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
        >
          Control Room
        </button>
      </div>

      {/* Right Cluster */}
      <div className="flex items-center gap-4">
        {/* Heartbeat & Sync Status */}
        <div className="hidden md:flex items-center gap-3 mr-4">
          <div className="flex flex-col items-end">
            <div className="flex items-center gap-1.5">
              <span className={`text-[10px] font-bold uppercase tracking-widest ${getSyncColor()}`}>
                {isStale ? 'Stale' : secondsSinceSync < 60 ? 'Live' : 'Sync OK'}
              </span>
              <div className={`w-1.5 h-1.5 rounded-full ${isStale ? 'bg-red-500' : 'bg-emerald-500'} animate-pulse shadow-[0_0_8px] shadow-emerald-500/50`} />
            </div>
            <span className="text-[9px] font-mono text-[var(--text-muted)] uppercase tracking-tighter">
              T+{secondsSinceSync}s
            </span>
          </div>
          <Activity className={`w-4 h-4 ${getSyncColor()} opacity-40`} />
        </div>

        <button 
          onClick={onThemeToggle}
          className="p-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <button className="p-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] relative">
          <Bell className="w-4 h-4" />
          <div className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-[var(--status-crimson)] rounded-full" />
        </button>

        <div className="h-6 w-px bg-[var(--divider)]" />

        <button onClick={onLogout} className="flex items-center gap-2 pl-2">
          <div className="w-7 h-7 rounded-sm bg-[var(--divider)] flex items-center justify-center">
            <User className="w-4 h-4 text-[var(--text-secondary)]" />
          </div>
        </button>
      </div>
    </div>
  );
}