import { ChevronRight, Bell, User, LogOut, ChevronDown } from 'lucide-react';

interface TopBarProps {
  breadcrumbs: string[];
  userName?: string;
  onLogout?: () => void;
}

export function TopBar({ breadcrumbs, userName = 'John Doe', onLogout }: TopBarProps) {
  return (
    <div 
      className="border-b px-8 flex items-center justify-between"
      style={{
        height: '64px',
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-default)'
      }}
    >
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2">
        {breadcrumbs.map((crumb, index) => (
          <div key={index} className="flex items-center gap-2">
            {index > 0 && (
              <ChevronRight className="w-4 h-4" style={{ color: '#94A3B8' }} />
            )}
            <span 
              className="text-sm"
              style={{
                color: index === breadcrumbs.length - 1 ? 'var(--text-primary)' : '#64748B',
                fontWeight: index === breadcrumbs.length - 1 ? 'var(--font-weight-semibold)' : 'var(--font-weight-regular)'
              }}
            >
              {crumb}
            </span>
          </div>
        ))}
      </div>

      {/* User Actions */}
      <div className="flex items-center gap-4">
        {/* Notifications */}
        <button 
          className="relative p-2 rounded-md transition-colors"
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          <Bell className="w-5 h-5" style={{ color: '#64748B' }} />
          <div 
            className="absolute top-1 right-1 w-4 h-4 rounded-full flex items-center justify-center text-xs"
            style={{ 
              backgroundColor: 'var(--danger-red)',
              color: 'white',
              fontSize: '10px',
              fontWeight: 'var(--font-weight-semibold)'
            }}
          >
            2
          </div>
        </button>

        {/* User Menu */}
        <div className="flex items-center gap-3">
          <span className="text-sm" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: 'var(--text-primary)'
          }}>
            {userName}
          </span>
          
          <button
            className="flex items-center gap-2 p-1 rounded-md transition-colors"
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <ChevronDown className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
          </button>
        </div>
      </div>
    </div>
  );
}