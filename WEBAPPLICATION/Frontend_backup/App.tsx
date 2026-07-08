import { useState } from 'react';
import { LoginPage } from './components/LoginPage';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewDashboard } from './components/OverviewDashboard';
import { LiveMonitor } from './components/LiveMonitor';
import { PlannerDashboard } from './components/PlannerDashboard';
import { ModelPerformance } from './components/ModelPerformance';
import { DataManagement } from './components/DataManagement';
import { ExplainabilityModal } from './components/ExplainabilityModal';

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [activeView, setActiveView] = useState('overview');
  const [showExplainabilityModal, setShowExplainabilityModal] = useState(false);
  const [userName, setUserName] = useState('John Doe');
  const [userRole, setUserRole] = useState('operator');

  const handleLogin = (role: string, name: string) => {
    setIsLoggedIn(true);
    setUserRole(role);
    setUserName(name);
    // Set default view based on role
    switch (role) {
      case 'analyst':
        setActiveView('model-performance');
        break;
      case 'planner':
        setActiveView('planner');
        break;
      case 'admin':
        setActiveView('settings');
        break;
      default:
        setActiveView('overview');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setActiveView('overview');
    setUserName('');
    setUserRole('');
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
      case 'settings':
        return <PlaceholderView title="Settings" />;
      default:
        return <OverviewDashboard />;
    }
  };

  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen" style={{ backgroundColor: 'var(--bg-primary)' }}>
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
        />

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto">
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
      <button
        onClick={() => setShowExplainabilityModal(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full flex items-center justify-center shadow-lg transition-all duration-200"
        style={{
          backgroundColor: 'var(--primary-blue)',
          color: 'white'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#1D4ED8';
          e.currentTarget.style.transform = 'scale(1.05)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
          e.currentTarget.style.transform = 'scale(1)';
        }}
        title="Why is this forecast high?"
      >
        <span className="text-xl">💡</span>
      </button>
    </div>
  );
}

interface PlaceholderViewProps {
  title: string;
}

function PlaceholderView({ title }: PlaceholderViewProps) {
  return (
    <div 
      className="rounded-lg p-12 text-center"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)'
      }}
    >
      <h2 
        className="text-2xl mb-2"
        style={{ 
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}
      >
        {title}
      </h2>
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
        This section is under development
      </p>
    </div>
  );
}