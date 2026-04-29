'use client';

import { AlertTriangle, Info, X, Loader2, AlertCircle, RefreshCcw, Clock, CheckCircle2 } from 'lucide-react';
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

  const alertConfig: Record<string, { icon: any, color: string, label: string }> = {
    critical: { icon: AlertCircle, color: 'var(--status-crimson)', label: 'CRITICAL' },
    warning: { icon: AlertTriangle, color: 'var(--status-amber)', label: 'WARNING' },
    info: { icon: Info, color: 'var(--brand-indigo)', label: 'INFO' },
    success: { icon: CheckCircle2, color: 'var(--status-emerald)', label: 'SOLVED' }
  };

  return (
    <div className="flex flex-col gap-4">
      
      {error && (
        <div className="px-4 py-2 bg-[var(--status-crimson)]/10 text-[var(--status-crimson)] text-[11px] font-bold uppercase tracking-widest rounded flex items-center gap-2">
          <AlertCircle className="w-4 h-4" /> {error}
        </div>
      )}

      <div className="flex items-center justify-between mb-2">
         <span className="micro-num text-[var(--text-muted)] uppercase tracking-widest">Live Alerts Grid</span>
         <button onClick={fetchAlerts} className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
            <RefreshCcw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
         </button>
      </div>

      <div className="space-y-3">
        {isLoading && alerts.length === 0 ? (
          <div className="py-12 flex flex-col items-center justify-center gap-3">
            <Loader2 className="w-6 h-6 animate-spin text-[var(--brand-indigo)] opacity-40" />
            <span className="micro-num animate-pulse uppercase">Syncing Alerts...</span>
          </div>
        ) : alerts.length === 0 ? (
          <div className="py-8 text-center glass-panel border-dashed">
             <span className="micro-num text-[var(--text-muted)] uppercase">All Systems Nominal</span>
          </div>
        ) : (
          alerts.map((alert) => {
            const config = alertConfig[alert.type] || alertConfig.info;
            const Icon = config.icon;

            return (
              <div key={alert.id} className="glass-panel p-4 flex flex-col gap-3 relative group">
                <button 
                  onClick={() => handleDismiss(alert.id)}
                  className="absolute top-4 right-4 text-[var(--text-muted)] hover:text-[var(--text-primary)] opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-3 h-3" />
                </button>

                <div className="flex items-center gap-3">
                  <div className={`p-1.5 rounded bg-[var(--surface-secondary)] border border-[var(--divider)] ${alert.type === 'critical' ? 'animate-pulse' : ''}`}>
                    <Icon className={`w-4 h-4`} style={{ color: config.color }} />
                  </div>
                  <div className="flex flex-col">
                     <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold tracking-tighter px-1 rounded bg-[var(--divider)]" style={{ color: config.color }}>{config.label}</span>
                        <h4 className="text-[13px] font-bold text-[var(--text-primary)]">{alert.title}</h4>
                     </div>
                  </div>
                </div>

                <p className="text-[12px] text-[var(--text-secondary)] leading-tight">{alert.detail}</p>

                <div className="flex items-center justify-between mt-1">
                   <div className="flex gap-2">
                      {alert.actions.map((action, i) => (
                        <button key={i} className={`px-2 py-1 rounded text-[10px] font-bold uppercase transition-all
                          ${action.primary 
                            ? 'bg-[var(--brand-indigo)] text-white' 
                            : 'bg-[var(--surface-secondary)] text-[var(--text-secondary)] border border-[var(--divider)] hover:text-[var(--text-primary)]'}
                        `}>
                          {action.label}
                        </button>
                      ))}
                   </div>
                   <div className="flex items-center gap-1.5 micro-num text-[var(--text-muted)]">
                      <Clock className="w-3 h-3" /> {alert.time}
                   </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      <button className="text-center w-full py-2 micro-num text-[var(--text-muted)] uppercase hover:text-[var(--text-primary)] transition-colors mt-2">
        View Archive Cluster
      </button>
    </div>
  );
}
