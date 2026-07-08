'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Sidebar } from '@/components/Sidebar';
import { TopBar } from '@/components/TopBar';
import { DigitalTwinDashboard } from '@/components/DigitalTwinDashboard';
import { DispatchSchedule } from '@/components/DispatchSchedule';
import { ForecastView } from '@/components/ForecastView';
import { DataManagement } from '@/components/DataManagement';
import { Settings } from '@/components/Settings';


function DashboardContent() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();

  // URL-driven state
  const activeView = searchParams.get('view') || 'overview';
  
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const setView = (view: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('view', view);
    router.replace(`${pathname}?${params.toString()}`);
  };

  const userName = user?.full_name || 'User';
  const userRole = user?.is_superuser ? 'ADMIN' : 'OPERATOR';

  const handleLogout = () => {
    logout();
  };

  const getBreadcrumbs = () => {
    switch (activeView) {
      case 'overview':
        return ['Control Room'];
      case 'forecast':
        return ['Forecasting', 'ECG Forecast'];
      case 'dispatch':
        return ['Dispatch Schedule'];
      case 'data-upload':
        return ['Data Management'];
      case 'settings':
        return ['Settings'];
      default:
        return ['Control Room'];
    }
  };

  const renderContent = () => {
    switch (activeView) {
      case 'overview':
        return <DigitalTwinDashboard />;
      case 'forecast':
        return <ForecastView />;
      case 'dispatch':
        return <DispatchSchedule />;
      case 'data-upload':
        return <DataManagement />;
      case 'settings':
        return <Settings />;
      default:
        return <DigitalTwinDashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-[var(--bg-page)] text-[var(--text-primary)] transition-colors duration-300 relative overflow-hidden">
        
      {/* Clean bento-style background — no grid, no glow */}

      <Sidebar
        activeView={activeView}
        onViewChange={setView}
        userName={userName}
        userRole={userRole}
      />

      <div className="flex-1 flex flex-col overflow-hidden relative z-10">
        <TopBar
          breadcrumbs={getBreadcrumbs()}
          userName={userName}
          onLogout={handleLogout}
          theme={theme}
          onThemeToggle={toggleTheme}
        />

        <main className="flex-1 overflow-y-auto p-8 relative">
          <div className="max-w-[1600px] mx-auto animate-in fade-in slide-in-from-bottom-2 duration-500">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <ProtectedRoute>
      <Suspense fallback={<div className="h-screen w-screen bg-black flex items-center justify-center text-white font-mono">INITIALIZING SYSTEM...</div>}>
        <DashboardContent />
      </Suspense>
    </ProtectedRoute>
  );
}
