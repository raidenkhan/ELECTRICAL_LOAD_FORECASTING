'use client';

import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Sidebar } from '@/components/Sidebar';
import { TopBar } from '@/components/TopBar';
import { OverviewDashboard } from '@/components/OverviewDashboard';
import { LiveMonitor } from '@/components/LiveMonitor';
import { PlannerDashboard } from '@/components/PlannerDashboard';
import { ModelPerformance } from '@/components/ModelPerformance';
import { DataManagement } from '@/components/DataManagement';
import { ExplainabilityView } from '@/components/ExplainabilityView';
import { ExplainabilityModal } from '@/components/ExplainabilityModal';
import { Settings } from '@/components/Settings';
import { useEffect } from 'react';


export default function Home() {
  const { user, logout } = useAuth();
  const [activeView, setActiveView] = useState('overview');
  const [showExplainabilityModal, setShowExplainabilityModal] = useState(false);
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



  // Derive user details from AuthContext
  const userName = user?.full_name || 'User';
  const userRole = user?.is_superuser ? 'admin' : 'operator';

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
        return <OverviewDashboard />;
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
    <ProtectedRoute>
      <div className="flex h-screen relative overflow-hidden" style={{ backgroundColor: 'var(--bg-page)', color: 'var(--text-primary)' }}>
        {/* Ambient Background Glows */}
        <div className="absolute top-[-10%] left-[-5%] w-[45%] h-[45%] rounded-full bg-primary/10 blur-[130px] pointer-events-none animate-pulse duration-[12s]" />
        <div className="absolute bottom-[0%] right-[-10%] w-[35%] h-[35%] rounded-full bg-primary/5 blur-[100px] pointer-events-none animate-pulse duration-[15s]" />
        <div className="absolute top-[25%] right-[15%] w-[25%] h-[25%] rounded-full bg-primary/5 blur-[90px] pointer-events-none" />
        <div className="absolute bottom-[20%] left-[10%] w-[15%] h-[15%] rounded-full bg-primary/10 blur-[70px] pointer-events-none" />



        {/* Sidebar */}
        <Sidebar
          activeView={activeView}
          onViewChange={setActiveView}
          userName={userName}
          userRole={userRole}
        />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Top Bar */}
          <TopBar
            breadcrumbs={getBreadcrumbs()}
            userName={userName}
            onLogout={handleLogout}
            theme={theme}
            onThemeToggle={toggleTheme}
          />


          {/* Content */}
          <main className="flex-1 overflow-y-auto p-6">
            <div className=" mx-auto">
              {renderContent()}
            </div>
          </main>
        </div>

        {/* Explainability Modal */}
        <ExplainabilityModal
          isOpen={showExplainabilityModal}
          onClose={() => setShowExplainabilityModal(false)}
        />

        {/* Floating Action Button for Demo */}
        {/* <button
          onClick={() => setShowExplainabilityModal(true)}
          className="fixed bottom-8 right-8 w-14 h-14 flex items-center justify-center transition-all duration-300 border border-white/10 glass-morphism group"
          style={{
            backgroundColor: 'var(--bg-surface)',
            backdropFilter: 'var(--glass-blur)',
            boxShadow: 'var(--glass-shadow)'
          }}
          title="Why is this forecast high?"
        >
          <div className="absolute inset-0 border border-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
          <span className="text-xl group-hover:scale-110 transition-transform">💡</span>
        </button> */}

      </div>
    </ProtectedRoute>
  );
}

interface PlaceholderViewProps {
  title: string;
}

function PlaceholderView({ title }: PlaceholderViewProps) {
  return (
    <div
      className="p-12 text-center border border-dashed glass-morphism"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border-primary)',
        backdropFilter: 'var(--glass-blur)'
      }}
    >
      <h2
        className="text-lg mb-2 uppercase tracking-widest font-bold"
        style={{
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-geist-mono)'
        }}
      >
        {title}
      </h2>
      <p className="text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
        This section is under development
      </p>
    </div>

  );
}
