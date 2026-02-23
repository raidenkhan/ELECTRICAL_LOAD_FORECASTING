import {
  LayoutDashboard,
  Activity,
  Upload,
  TrendingUp,
  Settings,
  ChevronRight,
  ChevronDown,
  Zap,
  User,
  Calendar
} from 'lucide-react';
import { useState } from 'react';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  userName: string;
  userRole?: string;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  hasSubmenu?: boolean;
  submenu?: { id: string; label: string }[];
}

export function Sidebar({ activeView, onViewChange, userName, userRole }: SidebarProps) {
  const [expandedItems, setExpandedItems] = useState<string[]>(['dashboard']);

  const navItems: NavItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <LayoutDashboard className="w-5 h-5" />,
      hasSubmenu: true,
      submenu: [
        { id: 'overview', label: 'Overview' },
        { id: 'live-monitor', label: 'Live Monitoring' }
      ]
    },
    {
      id: 'planner',
      label: 'Planning',
      icon: <Calendar className="w-5 h-5" />
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <TrendingUp className="w-5 h-5" />,
      hasSubmenu: true,
      submenu: [
        { id: 'model-performance', label: 'Model Performance' },
        { id: 'explainability', label: 'Explainability' }
      ]
    },
    {
      id: 'data-upload',
      label: 'Data Management',
      icon: <Upload className="w-5 h-5" />
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings className="w-5 h-5" />
    }
  ];

  const toggleExpanded = (id: string) => {
    setExpandedItems(prev =>
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  return (
    <div
      className="h-screen flex flex-col border-r"
      style={{
        width: '240px',
        backgroundColor: 'var(--bg-sidebar)',
        borderColor: 'var(--border-primary)'
      }}

    >
      {/* Logo Header */}
      <div
        className="px-6 flex flex-col justify-center border-b"
        style={{
          height: '72px',
          borderColor: 'var(--border-primary)'
        }}
      >
        <div className="flex items-center gap-3">
          <Zap className="w-5 h-5" style={{ color: 'var(--lime-primary)', filter: 'drop-shadow(0 0 8px var(--lime-primary))', opacity: 0.8 }} />
          <h2
            className="text-[13px] font-bold tracking-[0.2em] uppercase"
            style={{
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-geist-mono)'
            }}
          >
            GridForecast Pro
          </h2>
        </div>
        {userRole && (
          <p className="text-[10px] font-bold tracking-wider uppercase opacity-40 mt-1" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)', paddingLeft: '32px' }}>
            {userRole}
          </p>
        )}
      </div>


      {/* Navigation */}
      <nav className="flex-1 py-6 overflow-y-auto">
        <div className="space-y-1">
          {navItems.map((item) => {
            const isExpanded = expandedItems.includes(item.id);
            const isActive = activeView === item.id ||
              (item.submenu && item.submenu.some(sub => sub.id === activeView));

            return (
              <div key={item.id} className="relative">
                <button
                  onClick={() => {
                    if (item.hasSubmenu) {
                      toggleExpanded(item.id);
                    } else {
                      onViewChange(item.id);
                    }
                  }}
                  className="w-full flex items-center justify-between px-6 transition-all duration-200 group"
                  style={{
                    height: '44px',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-tertiary)',
                    backgroundColor: isActive ? 'rgba(255, 255, 255, 0.03)' : 'transparent',
                    borderLeft: isActive ? '2px solid var(--lime-primary)' : '2px solid transparent'
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div style={{ color: isActive ? 'var(--lime-primary)' : 'inherit' }}>
                      {item.icon}
                    </div>
                    <span className="text-[12px] font-bold tracking-tight uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                      {item.label}
                    </span>
                  </div>
                  {item.hasSubmenu && (
                    <div style={{ color: 'var(--text-muted)' }}>
                      {isExpanded ?
                        <ChevronDown className="w-4 h-4" /> :
                        <ChevronRight className="w-4 h-4" />
                      }
                    </div>
                  )}
                </button>

                {/* Submenu */}
                {item.hasSubmenu && item.submenu && isExpanded && (
                  <div className="mt-1 mb-2">
                    {item.submenu.map((subItem) => {
                      const isSubActive = activeView === subItem.id;

                      return (
                        <button
                          key={subItem.id}
                          onClick={() => onViewChange(subItem.id)}
                          className="w-full text-left flex items-center gap-3 transition-colors"
                          style={{
                            height: '36px',
                            paddingLeft: '48px',
                            paddingRight: '20px',
                            color: isSubActive ? 'var(--lime-primary)' : 'var(--text-tertiary)',
                            backgroundColor: 'transparent',
                            fontFamily: 'var(--font-geist-mono)'
                          }}
                        >
                          <div
                            className="w-1 h-1"
                            style={{ backgroundColor: isSubActive ? 'var(--lime-primary)' : 'transparent' }}
                          />
                          <span className="text-[11px] font-bold uppercase tracking-tight">{subItem.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </nav>


      {/* User Info */}
      <div
        className="px-6 border-t flex items-center gap-4"
        style={{
          height: '64px',
          backgroundColor: 'var(--hover-bg)',
          borderColor: 'var(--border-primary)'
        }}
      >

        <div
          className="w-9 h-9 border flex items-center justify-center flex-shrink-0"
          style={{ borderColor: 'var(--border-primary)', backgroundColor: 'var(--bg-page)', borderRadius: 0 }}
        >
          <User className="w-4 h-4" style={{ color: 'var(--text-tertiary)' }} />
        </div>


        <div className="flex-1 min-w-0">
          <p className="text-[12px] font-bold truncate tracking-tight uppercase" style={{
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-geist-mono)'
          }}>
            {userName}
          </p>
        </div>
        <ChevronRight className="w-4 h-4 flex-shrink-0 opacity-20" style={{ color: 'var(--text-muted)' }} />
      </div>

    </div>
  );
}