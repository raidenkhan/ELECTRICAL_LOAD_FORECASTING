'use client';

import { AlertTriangle, Info, CheckCircle, Clock, X } from 'lucide-react';
import { useState } from 'react';

interface Alert {
  id: number;
  type: 'critical' | 'warning' | 'info' | 'success';
  title: string;
  detail: string;
  time: string;
  actions: { label: string; primary?: boolean }[];
}

const alerts: Alert[] = [
  {
    id: 1,
    type: 'warning',
    title: 'Peak forecast exceeds capacity margin',
    detail: 'Expected: 1,580 MW at 14:30 | Available margin: 120 MW',
    time: '2 minutes ago',
    actions: [
      { label: 'View Details', primary: true },
      { label: 'Acknowledge' }
    ]
  },
  {
    id: 2,
    type: 'info',
    title: 'Data gap detected in feeder NY6ZA_LINE',
    detail: 'Using interpolated values for interval 13:00-13:15',
    time: '15 minutes ago',
    actions: [
      { label: 'View Impact', primary: true },
      { label: 'Dismiss' }
    ]
  },
  {
    id: 3,
    type: 'success',
    title: 'Model retraining completed',
    detail: 'New champion model v3.2 deployed with 96.8% accuracy',
    time: '1 hour ago',
    actions: [
      { label: 'View Report', primary: true }
    ]
  }
];

export function ActiveAlerts() {
  const [activeAlerts, setActiveAlerts] = useState(alerts);

  const handleDismiss = (id: number) => {
    setActiveAlerts(prev => prev.filter(a => a.id !== id));
  };

  const handleAction = (action: string, alertTitle: string) => {
    console.log(`Action: ${action} on ${alertTitle}`);
    // In a real app, this would trigger a mutation
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
      color: 'var(--status-warn)',
      bg: 'rgba(245, 158, 11, 0.08)',
      label: 'WARNING'
    },
    info: {
      icon: Info,
      color: 'var(--lime-primary)',
      bg: 'rgba(204, 255, 0, 0.05)',
      label: 'ADVISORY'
    },
    success: {
      icon: CheckCircle,
      color: 'var(--lime-primary)',
      bg: 'rgba(204, 255, 0, 0.08)',
      label: 'RESOLVED'
    }
  };



  return (
    <div
      className="border glass-morphism overflow-hidden relative"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border-primary)',
        backdropFilter: 'var(--glass-blur)',
        boxShadow: 'var(--glass-shadow)'
      }}
    >
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
        <button
          className="text-[11px] font-bold tracking-widest uppercase px-4 py-2 border transition-all hover:bg-white/5 active:scale-95"
          onClick={() => console.log('View Archive clicked')}
          style={{
            color: 'var(--text-secondary)',
            borderColor: 'var(--border-primary)',
            fontFamily: 'var(--font-geist-mono)'
          }}
        >
          View Archive
        </button>
      </div>


      {/* Alert Items */}
      <div className="divide-y" style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}>
        {activeAlerts.length === 0 ? (
          <div className="px-8 py-12 text-center text-sm text-gray-500">
            No active alerts
          </div>
        ) : (
          activeAlerts.map((alert) => {
            const config = alertConfig[alert.type];
            const Icon = config.icon;

            return (
              <div
                key={alert.id}
                className="px-8 py-6 border-b last:border-0 hover:bg-white/[0.02] transition-colors group"
              >

                <div className="flex items-start gap-4">
                  {/* Icon */}
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

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {/* Title Row */}
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
                        <h4 className="text-[12px] font-bold " style={{ color: 'var(--text-primary)' }}>
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

                    {/* Detail */}
                    <p className="text-sm mb-6 leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
                      {alert.detail}
                    </p>


                    {/* Footer */}
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


      {/* Footer */}
      <div
        className="px-6 py-4 border-t text-center"
        style={{
          borderColor: 'rgba(255, 255, 255, 0.05)',
          backgroundColor: 'rgba(255, 255, 255, 0.01)'
        }}
      >
        <button
          className="text-[11px] font-bold tracking-widest uppercase transition-colors"
          style={{ color: 'var(--lime-primary)', fontFamily: 'var(--font-geist-mono)' }}
        >
          System Notifications Preferences
        </button>
      </div>

    </div>
  );
}
