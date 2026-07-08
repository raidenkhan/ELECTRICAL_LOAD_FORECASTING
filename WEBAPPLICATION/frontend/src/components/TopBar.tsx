import { Bell, User, ChevronRight, Moon, Sun, Thermometer } from 'lucide-react';
import { useSystem } from '@/context/SystemContext';
import { useEffect, useState } from 'react';

interface TopBarProps {
  breadcrumbs: string[];
  userName?: string;
  onLogout?: () => void;
  theme?: string;
  onThemeToggle?: () => void;
}

export function TopBar({ 
  breadcrumbs, 
  onLogout, 
  theme, 
  onThemeToggle 
}: TopBarProps) {
  const { secondsSinceSync, isStale } = useSystem();
  const [currentTemp, setCurrentTemp] = useState<number | null>(null);

  useEffect(() => {
    const fetchTemp = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/forecast/dispatch/current-temp');
        if (res.ok) {
          const data = await res.json();
          setCurrentTemp(data.temperature_c);
        }
      } catch {
        // ignore
      }
    };
    fetchTemp();
    const interval = setInterval(fetchTemp, 5 * 60 * 1000); // every 5 min
    return () => clearInterval(interval);
  }, []);

  const getSyncColor = () => {
    if (isStale) return 'text-[var(--status-crimson)]';
    if (secondsSinceSync > 300) return 'text-[var(--status-amber)]';
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

      {/* Right Cluster */}
      <div className="flex items-center gap-4">
        {/* Sync Status + Temperature */}
        <div className="hidden md:flex items-center gap-3 mr-4">
          <div className="flex flex-col items-end">
            <div className="flex items-center gap-1.5">
              <span className={`text-[10px] font-bold uppercase tracking-widest ${getSyncColor()}`}>
                {isStale ? 'Stale' : secondsSinceSync < 60 ? 'Live' : 'Sync OK'}
              </span>
              <div className={`w-1.5 h-1.5 rounded-full ${isStale ? 'bg-red-500' : 'bg-emerald-500'} animate-pulse shadow-[0_0_8px] shadow-emerald-500/50`} />
            </div>
            <span className="text-[9px] font-mono text-[var(--text-muted)] uppercase tracking-tighter">
              {currentTemp !== null ? `${currentTemp.toFixed(1)}°C Accra` : '--°C'}
            </span>
          </div>
          <Thermometer className={`w-4 h-4 ${currentTemp !== null ? 'text-[var(--brand-blue)]' : 'text-[var(--text-muted)]'} opacity-60`} />
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
