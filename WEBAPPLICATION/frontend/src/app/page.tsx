'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Sidebar } from '@/components/Sidebar';
import { TopBar } from '@/components/TopBar';
import { OverviewDashboard } from '@/components/OverviewDashboard';
import { DigitalTwinDashboard } from '@/components/DigitalTwinDashboard';
import { LiveMonitor } from '@/components/LiveMonitor';
import { PlannerDashboard } from '@/components/PlannerDashboard';
import { ModelPerformance } from '@/components/ModelPerformance';
import { DataManagement } from '@/components/DataManagement';
import { ExplainabilityView } from '@/components/ExplainabilityView';
import { ExplainabilityModal } from '@/components/ExplainabilityModal';
import { Settings } from '@/components/Settings';

function DashboardContent() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();

  // URL-driven state
  const activeView = searchParams.get('view') || 'overview';
  const viewMode = (searchParams.get('mode') as 'analytics' | 'control-room') || 'control-room';
  
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
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

  const setViewMode = (mode: 'analytics' | 'control-room') => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('mode', mode);
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
        return ['Dashboard', 'Overview'];
      case 'live-monitor':
        return ['Dashboard', 'Live Monitor'];
      case 'planner':
        return ['Planning'];
      case 'data-upload':
        return ['Data Management'];
      case 'model-performance':
        return ['Analytics', 'Model Performance'];
      case 'explainability':
        return ['Analytics', 'Explainability'];
      case 'settings':
        return ['Settings'];
      default:
        return ['Dashboard'];
    }
  };

  const renderContent = () => {
    switch (activeView) {
      case 'overview':
        return viewMode === 'analytics' ? <OverviewDashboard /> : <DigitalTwinDashboard />;
      case 'live-monitor':
        return <LiveMonitor />;
      case 'planner':
        return <PlannerDashboard />;
      case 'data-upload':
        return <DataManagement />;
      case 'model-performance':
        return <ModelPerformance />;
      case 'explainability':
        return <ExplainabilityView />;
      case 'settings':
        return <Settings />;
      default:
        return <OverviewDashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-[var(--bg-page)] text-[var(--text-primary)] transition-colors duration-300 relative overflow-hidden">
        
      {/* Technical Ambient Background */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 opacity-[0.06] dark:opacity-[0.04]"
          style={{ 
            backgroundImage: `url('https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?q=80&w=2069&auto=format&fit=crop')`,
            backgroundSize: 'cover', backgroundPosition: 'center', filter: 'grayscale(100%) contrast(1.1)'
          }} />
        
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-[var(--brand-blue)]/30 dark:bg-[var(--brand-blue)]/20 blur-[150px] animate-pulse" style={{ animationDuration: '15s' }} />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[var(--brand-teal)]/25 dark:bg-[var(--brand-teal)]/15 blur-[150px] animate-pulse" style={{ animationDuration: '20s', animationDelay: '5s' }} />
        
        <div className="absolute inset-0 opacity-[0.04] dark:opacity-[0.05]"
          style={{ backgroundImage: 'radial-gradient(#6366F1 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
      </div>

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
          viewMode={activeView === 'overview' ? viewMode : undefined}
          onViewModeChange={setViewMode}
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
