import { ChevronRight, Bell, User, LogOut, ChevronDown, Sun, Moon } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';


interface TopBarProps {
  breadcrumbs: string[];
  userName?: string;
  onLogout?: () => void;
  theme?: string;
  onThemeToggle?: () => void;
}

export function TopBar({ breadcrumbs, userName = 'John Doe', onLogout, theme, onThemeToggle }: TopBarProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [notifications, setNotifications] = useState([
    { id: 1, text: 'System load peaking at 98%', time: '2m ago', read: false },
    { id: 2, text: 'New forecast model available', time: '1h ago', read: false },
    { id: 3, text: 'Database backup completed', time: '3h ago', read: true },
  ]);

  const notifRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifications(false);
      }
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleMarkAsRead = (id: number) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  return (
    <div
      className="px-8 flex items-center justify-between border-b glass-morphism z-40"
      style={{
        height: '72px',
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border-primary)',
        backdropFilter: 'var(--glass-blur)',
        boxShadow: 'var(--glass-shadow)',
        position: 'relative'
      }}
    >

      {/* Breadcrumbs */}
      <div className="flex items-center gap-3">
        {breadcrumbs.map((crumb, index) => (
          <div key={index} className="flex items-center gap-3">
            {index > 0 && (
              <ChevronRight className="w-3.5 h-3.5" style={{ color: 'var(--text-muted)' }} />
            )}
            <span
              className="text-[12px] font-bold uppercase tracking-widest"
              style={{
                color: index === breadcrumbs.length - 1 ? 'var(--text-primary)' : 'var(--text-tertiary)',
                fontFamily: 'var(--font-geist-mono)'
              }}
            >
              {crumb}
            </span>
          </div>
        ))}
      </div>


      {/* User Actions */}
      <div className="flex items-center gap-4">
        {/* Theme Toggle */}
        <button
          onClick={onThemeToggle}
          className="w-9 h-9 border flex items-center justify-center transition-colors hover:bg-white/5 active:scale-95"
          style={{
            borderColor: 'var(--border-primary)',
            color: 'var(--text-tertiary)',
            borderRadius: 0
          }}
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        {/* Notifications */}
        <div className="relative" ref={notifRef}>
          <button
            className="w-9 h-9 border flex items-center justify-center transition-colors hover:bg-white/5 active:scale-95 relative"
            style={{ borderColor: 'var(--border-primary)', borderRadius: 0 }}
            onClick={() => setShowNotifications(!showNotifications)}
          >
            <Bell className="w-4 h-4" style={{ color: 'var(--text-tertiary)' }} />
            {unreadCount > 0 && (
              <div
                className="absolute -top-1 -right-1 w-4 h-4 flex items-center justify-center text-[10px] font-bold"
                style={{
                  backgroundColor: 'var(--status-error)',
                  color: '#000',
                  fontFamily: 'var(--font-geist-mono)'
                }}
              >
                {unreadCount}
              </div>
            )}
          </button>

          {/* Notifications Dropdown */}
          {showNotifications && (
            <div className="absolute top-12 right-0 w-80 border glass-morphism z-50 animate-in fade-in slide-in-from-top-2"
              style={{
                backgroundColor: 'var(--bg-surface)',
                borderColor: 'var(--border-primary)',
                boxShadow: 'var(--glass-shadow)'
              }}>
              <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border-primary)' }}>
                <h3 className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>Notifications</h3>
                <button onClick={() => setNotifications(prev => prev.map(n => ({ ...n, read: true })))} className="text-[10px] uppercase hover:underline" style={{ color: 'var(--lime-primary)' }}>Mark all read</button>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {notifications.length === 0 ? (
                  <div className="p-8 text-center text-xs opacity-50">No new notifications</div>
                ) : (
                  notifications.map(n => (
                    <div key={n.id} onClick={() => handleMarkAsRead(n.id)} className="p-4 border-b last:border-0 hover:bg-white/5 cursor-pointer transition-colors relative" style={{ borderColor: 'var(--border-primary)', opacity: n.read ? 0.5 : 1 }}>
                      {!n.read && <div className="absolute top-4 left-2 w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--lime-primary)' }} />}
                      <p className="text-xs font-medium mb-1 pl-2" style={{ color: 'var(--text-primary)' }}>{n.text}</p>
                      <p className="text-[10px] opacity-50 pl-2 uppercase tracking-wide" style={{ fontFamily: 'var(--font-geist-mono)' }}>{n.time}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* User Menu */}
        <div className="relative pl-2 border-l" style={{ borderColor: 'var(--border-primary)' }} ref={userMenuRef}>
          <div
            className="flex items-center gap-4 cursor-pointer hover:bg-white/5 p-1 transition-colors"
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div className="text-right hidden md:block">
              <p className="text-[11px] font-bold uppercase tracking-tight" style={{
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-geist-mono)'
              }}>
                {userName}
              </p>
              <p className="text-[9px] font-bold uppercase tracking-widest opacity-40 ml-auto" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
                Online
              </p>
            </div>

            <div
              className="w-9 h-9 border flex items-center justify-center transition-transform duration-200"
              style={{
                borderColor: 'var(--border-primary)',
                borderRadius: 0,
                transform: showUserMenu ? 'rotate(180deg)' : 'none'
              }}
            >
              <ChevronDown className="w-3.5 h-3.5" style={{ color: 'var(--text-muted)' }} />
            </div>
          </div>

          {/* User Dropdown */}
          {showUserMenu && (
            <div className="absolute top-12 right-0 w-48 border glass-morphism z-50 animate-in fade-in slide-in-from-top-2"
              style={{
                backgroundColor: 'var(--bg-surface)',
                borderColor: 'var(--border-primary)',
                boxShadow: 'var(--glass-shadow)'
              }}>
              <div className="py-1">
                <button className="w-full text-left px-4 py-3 text-xs font-bold uppercase tracking-wider hover:bg-white/5 transition-colors flex items-center gap-3" style={{ color: 'var(--text-secondary)' }}>
                  <User className="w-3.5 h-3.5" /> Profile
                </button>
                <button
                  className="w-full text-left px-4 py-3 text-xs font-bold uppercase tracking-wider hover:bg-white/5 transition-colors flex items-center gap-3 border-t"
                  style={{ color: 'var(--status-error)', borderColor: 'var(--border-primary)' }}
                  onClick={onLogout}
                >
                  <LogOut className="w-3.5 h-3.5" /> Sign Out
                </button>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}