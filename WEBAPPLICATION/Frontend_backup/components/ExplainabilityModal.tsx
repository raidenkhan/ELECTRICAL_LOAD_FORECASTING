import { X, AlertTriangle } from 'lucide-react';

interface ExplainabilityModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ExplainabilityModal({ isOpen, onClose }: ExplainabilityModalProps) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 flex items-center justify-center z-50 p-4"
      style={{
        backgroundColor: 'rgba(0,0,0,0.4)',
        backdropFilter: 'blur(4px)'
      }}
      onClick={onClose}
    >
      <div 
        className="rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          boxShadow: '0 20px 25px rgba(0,0,0,0.15)',
          padding: '32px'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div className="flex-1">
            <h2 className="text-xl mb-2" style={{ 
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--text-primary)'
            }}>
              Why is this forecast high?
            </h2>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Forecast: Feb 4, 14:30 | 102 MW ± 8 MW
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-md transition-colors"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Sections */}
        <div className="space-y-6">
          {/* Section 1: Regime Context */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">1️⃣</span>
              <h3 className="text-base" style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                Regime Context
              </h3>
            </div>
            <div 
              className="rounded-lg p-4"
              style={{
                backgroundColor: '#EFF6FF',
                border: '1px solid var(--primary-blue)'
              }}
            >
              <p className="text-sm mb-2" style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                You are in Regime 2: Seasonal High
              </p>
              <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
                Typical for afternoon peaks in winter
              </p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Historical range: 95-110 MW
              </p>
            </div>
          </div>

          {/* Section 2: Top Drivers */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">2️⃣</span>
              <h3 className="text-base" style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                Top Drivers
              </h3>
            </div>
            <div 
              className="rounded-lg p-4 space-y-2"
              style={{
                backgroundColor: '#F8FAFC',
                border: '1px solid var(--border-default)'
              }}
            >
              <DriverItem text="Yesterday's load (Lag_96): +12 MW" />
              <DriverItem text="Grid inflow (NY6ZA): +8 MW" />
              <DriverItem text="Time of day (Hour 14): +5 MW" />
              <button
                className="text-sm mt-3 px-3 py-1.5 rounded transition-colors"
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
                View Full SHAP Analysis →
              </button>
            </div>
          </div>

          {/* Section 3: Confidence */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">3️⃣</span>
              <h3 className="text-base flex items-center gap-2" style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                Confidence: 
                <span className="flex items-center gap-1" style={{ color: 'var(--warning-orange)' }}>
                  MEDIUM <AlertTriangle className="w-4 h-4" />
                </span>
              </h3>
            </div>
            <div 
              className="rounded-lg p-4"
              style={{
                backgroundColor: '#FFFBEB',
                border: '1px solid var(--warning-orange)'
              }}
            >
              <p className="text-sm mb-2" style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}>
                Reason: Historical variance ±8 MW in this hour
              </p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Recommendation: Monitor closely, have reserves ready
              </p>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-end gap-3 mt-8 pt-6" style={{ borderTop: '1px solid var(--border-default)' }}>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-md text-sm transition-colors"
            style={{
              border: '1px solid var(--border-default)',
              color: 'var(--text-secondary)',
              fontWeight: 'var(--font-weight-medium)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            Close
          </button>
          <button
            className="px-4 py-2 rounded-md text-sm transition-colors"
            style={{
              backgroundColor: 'var(--primary-blue)',
              color: 'white',
              fontWeight: 'var(--font-weight-medium)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#1D4ED8';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
            }}
          >
            Export Report
          </button>
        </div>
      </div>
    </div>
  );
}

function DriverItem({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>•</span>
      <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{text}</span>
    </div>
  );
}
