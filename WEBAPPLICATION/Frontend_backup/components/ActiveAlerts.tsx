import { AlertTriangle, Info, CheckCircle } from 'lucide-react';

interface Alert {
  id: number;
  type: 'warning' | 'info' | 'success';
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
    detail: 'Expected: 102 MW at 14:30 | Margin: 5 MW',
    time: '2m ago',
    actions: [
      { label: 'View Details', primary: true },
      { label: 'Dismiss' }
    ]
  },
  {
    id: 2,
    type: 'info',
    title: 'NY6ZA_LINE data gap detected',
    detail: 'Using interpolated values (13:00-13:15)',
    time: '15m ago',
    actions: [
      { label: 'View Impact', primary: true },
      { label: 'Dismiss' }
    ]
  },
  {
    id: 3,
    type: 'success',
    title: 'Model retraining completed successfully',
    detail: 'New champion model v3.2 deployed',
    time: '1h ago',
    actions: [
      { label: 'View Report', primary: true },
      { label: 'Dismiss' }
    ]
  }
];

export function ActiveAlerts() {
  const alertConfig = {
    warning: {
      icon: AlertTriangle,
      borderColor: 'var(--warning-orange)',
      backgroundColor: '#FFFBEB',
      iconColor: 'var(--warning-orange)'
    },
    info: {
      icon: Info,
      borderColor: 'var(--primary-blue)',
      backgroundColor: '#EFF6FF',
      iconColor: 'var(--primary-blue)'
    },
    success: {
      icon: CheckCircle,
      borderColor: 'var(--success-green)',
      backgroundColor: '#ECFDF5',
      iconColor: 'var(--success-green)'
    }
  };

  return (
    <div 
      className="rounded-lg"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
      }}
    >
      {/* Header */}
      <div 
        className="flex items-center justify-between px-6 py-4 border-b"
        style={{ borderColor: 'var(--border-default)' }}
      >
        <h3 style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Active Alerts ({alerts.length})
        </h3>
        <button 
          className="text-sm px-3 py-1.5 rounded-md transition-colors"
          style={{
            color: 'var(--primary-blue)',
            fontWeight: 'var(--font-weight-medium)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#EFF6FF';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          View All
        </button>
      </div>

      {/* Alert Items */}
      <div className="p-6 space-y-3">
        {alerts.map((alert) => {
          const config = alertConfig[alert.type];
          const Icon = config.icon;

          return (
            <div
              key={alert.id}
              className="rounded-md p-4"
              style={{
                backgroundColor: config.backgroundColor,
                borderLeft: `4px solid ${config.borderColor}`
              }}
            >
              <div className="flex items-start gap-3">
                <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: config.iconColor }} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm mb-1" style={{ 
                    fontWeight: 'var(--font-weight-semibold)',
                    color: 'var(--text-primary)'
                  }}>
                    {alert.title}
                  </p>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>
                    {alert.detail}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {alert.actions.map((action, index) => (
                        <button
                          key={index}
                          className="text-xs px-3 py-1.5 rounded transition-colors"
                          style={{
                            backgroundColor: action.primary ? config.borderColor : 'transparent',
                            color: action.primary ? 'white' : config.borderColor,
                            fontWeight: 'var(--font-weight-medium)',
                            border: action.primary ? 'none' : `1px solid ${config.borderColor}`
                          }}
                          onMouseEnter={(e) => {
                            if (!action.primary) {
                              e.currentTarget.style.backgroundColor = `${config.borderColor}15`;
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (!action.primary) {
                              e.currentTarget.style.backgroundColor = 'transparent';
                            }
                          }}
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {alert.time}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
