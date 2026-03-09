'use client';

import { AlertTriangle, Info, CheckCircle, Clock, X, Loader2, AlertCircle, RefreshCcw, Download, CheckCircle2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { forecastService } from '../services/forecastService';

interface Alert {
  id: number;
  type: 'critical' | 'warning' | 'info' | 'success';
  title: string;
  detail: string;
  time: string;
  actions: { label: string; primary?: boolean }[];
}

export function ActiveAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = async () => {
    try {
      const liveAlerts = await forecastService.getAlerts();
      setAlerts(liveAlerts);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
      setError('Connection lost to alerts feed');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 60000); // Poll every 60s
    return () => clearInterval(interval);
  }, []);

  const handleDismiss = (id: number) => {
    setAlerts(prev => prev.filter(a => a.id !== id));
  };

  const handleAction = (action: string, alertTitle: string) => {
    console.log(`Action: ${action} on ${alertTitle}`);
  };

  const alertConfig = {
    critical: {
      icon: AlertTriangle,
      color: 'var(--status-error)',
      bg: 'rgba(239, 68, 68, 0.08)',
      label: 'CRITICAL'
    },
    warning: {
      icon: AlertTriangle,
      color: 'var(--status-warning)',
      bg: 'rgba(245, 158, 11, 0.08)',
      label: 'WARNING'
    },
    info: {
      icon: Info,
      color: 'var(--text-secondary)',
      bg: 'rgba(255, 255, 255, 0.03)',
      label: 'INFO'
    },
    success: {
      icon: CheckCircle2,
      color: 'var(--status-ok)',
      bg: 'rgba(16, 185, 129, 0.08)',
      label: 'RESOLVED'
    }
  };

  return (
    <div className="border glass-morphism overflow-hidden relative"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border-primary)',
        backdropFilter: 'var(--glass-blur)',
        boxShadow: 'var(--glass-shadow)'
      }}>

      {/* Decorative Corner */}
      <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden pointer-events-none opacity-40">
        <div className="absolute top-0 right-0 w-[1px] h-4 bg-white" />
        <div className="absolute top-0 right-0 w-4 h-[1px] bg-white" />
      </div>

      <div
        className="flex items-center justify-between px-8 py-6 border-b"
        style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}
      >
        <div className="flex items-center gap-4">
          <div className="w-2 h-5" style={{ backgroundColor: 'var(--lime-primary)' }} />
          <h3 className="text-sm font-bold tracking-[0.3em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
            Active Alerts Grid
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchAlerts}
            disabled={isLoading}
            className="p-2 border border-border hover:bg-white/5 disabled:opacity-50"
            title="Refresh feed"
          >
            <RefreshCcw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            className="text-[11px] font-bold tracking-widest uppercase px-4 py-2 border transition-all hover:bg-white/5"
            style={{
              color: 'var(--text-secondary)',
              borderColor: 'var(--border-primary)',
              fontFamily: 'var(--font-geist-mono)'
            }}
          >
            View Archive
          </button>
        </div>
      </div>

      <div className="divide-y" style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}>
        {error && (
          <div className="px-8 py-4 bg-red-500/5 text-red-500 text-[10px] font-black uppercase tracking-widest flex items-center gap-2">
            <AlertCircle className="w-4 h-4" /> {error}
          </div>
        )}

        {isLoading && alerts.length === 0 ? (
          <div className="px-8 py-12 flex flex-col items-center justify-center space-y-4">
            <Loader2 className="w-8 h-8 animate-spin text-lime-primary opacity-20" />
            <span className="text-[10px] font-black tracking-widest text-muted-foreground uppercase">Syncing Alert Feed...</span>
          </div>
        ) : alerts.length === 0 ? (
          <div className="px-8 py-12 flex flex-col items-center justify-center space-y-4">
            <CheckCircle2 className="w-8 h-8 text-emerald-500 opacity-20" />
            <span className="text-[10px] font-black tracking-widest text-muted-foreground uppercase">All Systems Optimal</span>
          </div>
        ) : (
          alerts.map((alert) => {
            const config = alertConfig[alert.type as keyof typeof alertConfig] || alertConfig.info;
            const Icon = config.icon;

            return (
              <div
                key={alert.id}
                className="px-8 py-6 hover:bg-white/[0.02] transition-colors group"
              >
                <div className="flex items-start gap-4">
                  <div
                    className="w-9 h-9 flex items-center justify-center flex-shrink-0 border"
                    style={{
                      backgroundColor: config.bg,
                      borderColor: `${config.color}33`,
                      color: config.color
                    }}
                  >
                    <Icon className="w-4 h-4" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4 mb-1.5">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span
                          className="text-[10px] font-bold px-1.5 py-0.5 border"
                          style={{
                            borderColor: `${config.color}33`,
                            color: config.color,
                            fontFamily: 'var(--font-geist-mono)'
                          }}
                        >
                          {config.label}
                        </span>
                        <h4 className="text-[12px] font-bold" style={{ color: 'var(--text-primary)' }}>
                          {alert.title}
                        </h4>
                      </div>
                      <button
                        className="p-1 hover:bg-white/5 transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100"
                        title="Dismiss"
                        onClick={() => handleDismiss(alert.id)}
                      >
                        <X className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                      </button>
                    </div>

                    <p className="text-sm mb-6 leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
                      {alert.detail}
                    </p>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {alert.actions.map((action, index) => (
                          <button
                            key={index}
                            onClick={() => handleAction(action.label, alert.title)}
                            className="text-[11px] px-3 py-1.5 transition-all font-bold uppercase tracking-wider border active:scale-95"
                            style={{
                              backgroundColor: action.primary ? config.color : 'transparent',
                              color: action.primary ? '#000' : 'var(--text-secondary)',
                              borderColor: action.primary ? config.color : 'var(--border-primary)',
                              fontFamily: 'var(--font-geist-mono)'
                            }}
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                      <div className="flex items-center gap-1.5 text-[11px] font-bold" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-geist-mono)' }}>
                        <Clock className="w-3.5 h-3.5" />
                        {alert.time.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      <div
        className="px-6 py-4 border-t text-center"
        style={{
          borderColor: 'rgba(255, 255, 255, 0.05)',
          backgroundColor: 'rgba(255, 255, 255, 0.01)'
        }}
      >
        <button
          className="text-[11px] font-bold tracking-widest uppercase transition-colors text-primary hover:text-primary-light"
          style={{ fontFamily: 'var(--font-geist-mono)' }}
        >
          System Notifications Preferences
        </button>
      </div>
    </div>
  );
}
