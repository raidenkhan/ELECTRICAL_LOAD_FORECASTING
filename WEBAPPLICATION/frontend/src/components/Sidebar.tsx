import {
  LayoutDashboard,
  Calendar,
  TrendingUp,
  Upload,
  Settings,
  Activity,
  Zap,
  CheckCircle2
} from 'lucide-react';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  userName: string;
  userRole?: string;
}

export function Sidebar({ activeView, onViewChange, userRole }: SidebarProps) {
  const navGroups = [
    {
      label: 'Dashboard',
      items: [
        { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="w-4 h-4" /> },
        { id: 'live-monitor', label: 'Live Monitoring', icon: <Activity className="w-4 h-4" /> }
      ]
    },
    {
      label: 'Planning',
      items: [
        { id: 'planner', label: 'Planning', icon: <Calendar className="w-4 h-4" /> }
      ]
    },
    {
      label: 'Analytics',
      items: [
        { id: 'model-performance', label: 'Model Performance', icon: <TrendingUp className="w-4 h-4" /> },
        { id: 'explainability', label: 'Explainability', icon: <Zap className="w-4 h-4" /> }
      ]
    },
    {
      label: 'System',
      items: [
        { id: 'data-upload', label: 'Data Management', icon: <Upload className="w-4 h-4" /> },
        { id: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> }
      ]
    }
  ];

  return (
    <div className="h-screen w-[200px] flex flex-col border-r bg-white/40 dark:bg-gray-900/40 backdrop-blur-2xl border-white/10 dark:border-white/5 z-50">
      {/* Brand Header */}
      <div className="px-5 py-6 flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-[var(--brand-indigo)] fill-[var(--brand-indigo)]" />
          <h1 className="text-[14px] font-bold tracking-tight text-[var(--text-primary)]">
            GRIDCo Forecast Pro
          </h1>
        </div>
        <div className="px-2 py-0.5 bg-[var(--divider)] rounded-sm w-fit mt-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-secondary)]">
            {userRole || 'OPERATOR'}
          </span>
        </div>
      </div>

      <div className="h-px bg-[var(--divider)] mx-5" />

      {/* Navigation Groups */}
      <nav className="flex-1 py-6 px-3 space-y-6 overflow-y-auto">
        {navGroups.map((group) => (
          <div key={group.label} className="space-y-1">
            <h3 className="px-2 text-[11px] font-bold uppercase tracking-widest text-[var(--text-muted)] mb-2">
              {group.label}
            </h3>
            {group.items.map((item) => {
              const isActive = activeView === item.id;
              return (
                <div key={item.id} className="border-b border-[var(--divider)]/30 last:border-0">
                  <button
                    onClick={() => onViewChange(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 transition-all duration-200 group relative
                      ${isActive
                        ? 'bg-[var(--brand-blue)]/10 text-[var(--brand-blue)] border-l-2 border-[var(--brand-blue)]'
                        : 'text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)]/50 hover:text-[var(--text-primary)]'
                      }`}
                  >
                    <div className={`${isActive ? 'text-[var(--brand-blue)]' : 'group-hover:text-[var(--text-primary)] transition-colors'}`}>
                      {item.icon}
                    </div>
                    <span className="text-[13px] font-semibold" style={{ fontWeight: 600 }}>{item.label}</span>
                  </button>
                </div>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer Status */}
      <div className="p-4 bg-[var(--surface-secondary)] border-t border-[var(--divider)]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--status-emerald)] animate-pulse" />
          <span className="text-[12px] font-medium text-[var(--text-primary)]">System Online</span>
        </div>
        <p className="text-[10px] text-[var(--text-muted)] mt-1 font-mono">
          Last sync: 14:20
        </p>
      </div>
    </div>
  );
}
