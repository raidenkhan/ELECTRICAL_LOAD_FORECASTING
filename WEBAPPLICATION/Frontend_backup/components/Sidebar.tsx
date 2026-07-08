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
        { id: 'model-performance', label: 'Model Performance' }
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
        borderColor: 'var(--border-default)'
      }}
    >
      {/* Logo Header */}
      <div 
        className="px-5 border-b flex flex-col justify-center"
        style={{ 
          height: '64px',
          borderColor: 'var(--border-default)'
        }}
      >
        <div className="flex items-center gap-2 mb-0.5">
          <Zap className="w-6 h-6" style={{ color: 'var(--primary-blue)' }} />
          <h2 
            className="text-base"
            style={{ 
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--text-primary)'
            }}
          >
            Grid Forecast Pro
          </h2>
        </div>
        {userRole && (
          <p className="text-xs ml-8 capitalize" style={{ color: '#64748B' }}>
            {userRole}
          </p>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-5 overflow-y-auto">
        <div className="space-y-0.5">
          {navItems.map((item) => {
            const isExpanded = expandedItems.includes(item.id);
            const isActive = activeView === item.id || 
              (item.submenu && item.submenu.some(sub => sub.id === activeView));

            return (
              <div key={item.id}>
                <button
                  onClick={() => {
                    if (item.hasSubmenu) {
                      toggleExpanded(item.id);
                    } else {
                      onViewChange(item.id);
                    }
                  }}
                  className="w-full flex items-center justify-between px-5 transition-colors"
                  style={{
                    height: '40px',
                    color: 'var(--text-secondary)',
                    backgroundColor: 'transparent'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#F8FAFC';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <div className="flex items-center gap-3">
                    {item.icon}
                    <span className="text-sm" style={{ fontWeight: 'var(--font-weight-medium)' }}>
                      {item.label}
                    </span>
                  </div>
                  {item.hasSubmenu && (
                    isExpanded ? 
                      <ChevronDown className="w-4 h-4" /> : 
                      <ChevronRight className="w-4 h-4" />
                  )}
                  {!item.hasSubmenu && (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
                
                {/* Submenu */}
                {item.hasSubmenu && item.submenu && isExpanded && (
                  <div className="mt-0.5 mb-0.5">
                    {item.submenu.map((subItem) => {
                      const isSubActive = activeView === subItem.id;
                      
                      return (
                        <button
                          key={subItem.id}
                          onClick={() => onViewChange(subItem.id)}
                          className="w-full text-left flex items-center gap-2 transition-colors"
                          style={{
                            height: '40px',
                            paddingLeft: '40px',
                            paddingRight: '20px',
                            color: isSubActive ? 'var(--primary-blue)' : '#64748B',
                            backgroundColor: isSubActive ? '#EFF6FF' : 'transparent',
                            fontWeight: isSubActive ? 'var(--font-weight-medium)' : 'var(--font-weight-regular)'
                          }}
                          onMouseEnter={(e) => {
                            if (!isSubActive) {
                              e.currentTarget.style.backgroundColor = '#F8FAFC';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (!isSubActive) {
                              e.currentTarget.style.backgroundColor = 'transparent';
                            }
                          }}
                        >
                          {isSubActive && (
                            <div 
                              className="w-1.5 h-1.5 rounded-full"
                              style={{ backgroundColor: 'var(--primary-blue)' }}
                            />
                          )}
                          {!isSubActive && <div className="w-1.5" />}
                          <span className="text-sm">{subItem.label}</span>
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
        className="px-5 border-t flex items-center gap-3"
        style={{ 
          height: '56px',
          backgroundColor: '#F8FAFC',
          borderColor: 'var(--border-default)'
        }}
      >
        <div 
          className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: 'var(--primary-blue)' }}
        >
          <User className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm truncate" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: 'var(--text-primary)'
          }}>
            {userName}
          </p>
        </div>
        <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
      </div>
    </div>
  );
}