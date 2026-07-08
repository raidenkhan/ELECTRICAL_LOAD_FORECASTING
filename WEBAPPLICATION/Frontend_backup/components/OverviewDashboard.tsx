import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Zap, 
  AlertTriangle,
  CheckCircle,
  HelpCircle
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ForecastChart } from './ForecastChart';
import { ActiveAlerts } from './ActiveAlerts';
import { useState } from 'react';
import { ExplainabilityModal } from './ExplainabilityModal';

// Mock data for charts
const forecastData = [
  { time: '00:00', actual: 1200, forecast: 1180, upper: 1250, lower: 1150 },
  { time: '04:00', actual: 950, forecast: 980, upper: 1050, lower: 920 },
  { time: '08:00', actual: 1450, forecast: 1420, upper: 1500, lower: 1380 },
  { time: '12:00', actual: 1650, forecast: 1680, upper: 1750, lower: 1620 },
  { time: '16:00', actual: 1580, forecast: 1550, upper: 1620, lower: 1500 },
  { time: '20:00', actual: 1380, forecast: 1400, upper: 1480, lower: 1350 },
  { time: '24:00', actual: 1100, forecast: 1120, upper: 1180, lower: 1080 }
];

const regimeData = [
  { hour: '00-04', regime0: 60, regime1: 25, regime2: 15 },
  { hour: '04-08', regime0: 45, regime1: 35, regime2: 20 },
  { hour: '08-12', regime0: 30, regime1: 40, regime2: 30 },
  { hour: '12-16', regime0: 25, regime1: 35, regime2: 40 },
  { hour: '16-20', regime0: 35, regime1: 40, regime2: 25 },
  { hour: '20-24', regime0: 55, regime1: 30, regime2: 15 }
];

export function OverviewDashboard() {
  const [showExplainability, setShowExplainability] = useState(false);

  return (
    <div className="space-y-6">
      {/* Hero Metrics - 3 Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <HeroMetricCard
          label="CURRENT LOAD"
          value="87.3 MW"
          trend="+2.4%"
          trendDirection="up"
          isLive={true}
        />
        <HeroMetricCard
          label="NEXT PEAK"
          value="14:30 Today"
          subtitle="102 MW"
          detail="± 8 MW"
          onExplain={() => setShowExplainability(true)}
        />
        <HeroMetricCard
          label="RISK LEVEL"
          value="MEDIUM"
          subtitle="Regime 1"
          riskLevel="medium"
        />
      </div>

      {/* Forecast Chart */}
      <ForecastChart />

      {/* Active Alerts */}
      <ActiveAlerts />

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regime Probability */}
        <div 
          className="rounded-lg p-6"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
          }}
        >
          <h3 
            className="text-lg mb-1"
            style={{ 
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--text-primary)'
            }}
          >
            Regime Probability Distribution
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
            Operating regime forecast by time period
          </p>
          
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={regimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
              <XAxis 
                dataKey="hour" 
                stroke="var(--text-muted)"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="var(--text-muted)"
                style={{ fontSize: '12px' }}
                label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-default)',
                  borderRadius: '6px',
                  fontSize: '12px'
                }}
              />
              <Legend />
              <Bar dataKey="regime0" stackId="a" fill="var(--regime-0)" name="Standard Operation" />
              <Bar dataKey="regime1" stackId="a" fill="var(--regime-1)" name="Transition/Chaos" />
              <Bar dataKey="regime2" stackId="a" fill="var(--regime-2)" name="Seasonal High" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* System Status */}
        <div 
          className="rounded-lg p-6"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
          }}
        >
          <h3 
            className="text-lg mb-1"
            style={{ 
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--text-primary)'
            }}
          >
            System Status
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
            Real-time monitoring and alerts
          </p>

          <div className="space-y-4">
            {/* Status Items */}
            <StatusItem 
              label="Data Feed"
              status="connected"
              detail="Last update: 2 min ago"
            />
            <StatusItem 
              label="Model Engine"
              status="connected"
              detail="Running: LSTM v2.3"
            />
            <StatusItem 
              label="Forecast Generation"
              status="connected"
              detail="Next cycle: 15 min"
            />
            <StatusItem 
              label="API Gateway"
              status="degraded"
              detail="Elevated latency detected"
            />
          </div>
        </div>
      </div>

      {/* Explainability Modal */}
      <ExplainabilityModal 
        isOpen={showExplainability}
        onClose={() => setShowExplainability(false)}
      />
    </div>
  );
}

interface HeroMetricCardProps {
  label: string;
  value: string;
  subtitle?: string;
  detail?: string;
  trend?: string;
  trendDirection?: 'up' | 'down';
  isLive?: boolean;
  riskLevel?: 'low' | 'medium' | 'high';
  onExplain?: () => void;
}

function HeroMetricCard({ 
  label, 
  value, 
  subtitle, 
  detail, 
  trend, 
  trendDirection, 
  isLive,
  riskLevel,
  onExplain
}: HeroMetricCardProps) {
  const TrendIcon = trendDirection === 'up' ? TrendingUp : TrendingDown;
  
  const riskConfig = {
    low: { bg: '#DCFCE7', text: '#059669' },
    medium: { bg: '#FEF3C7', text: '#F59E0B' },
    high: { bg: '#FEE2E2', text: '#DC2626' }
  };

  return (
    <div 
      className="rounded-lg relative"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        padding: '24px',
        height: '140px',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Explain Button */}
      {onExplain && (
        <button
          onClick={onExplain}
          className="absolute top-3 right-3 w-7 h-7 rounded-full flex items-center justify-center transition-all"
          style={{
            backgroundColor: '#EFF6FF',
            color: 'var(--primary-blue)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
            e.currentTarget.style.color = 'white';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#EFF6FF';
            e.currentTarget.style.color = 'var(--primary-blue)';
          }}
          title="Why is this forecast high?"
        >
          <HelpCircle className="w-4 h-4" />
        </button>
      )}

      {/* Card Header */}
      <div className="text-xs mb-3" style={{ 
        fontWeight: 'var(--font-weight-semibold)',
        color: '#64748B',
        letterSpacing: '0.5px',
        textTransform: 'uppercase'
      }}>
        {label}
      </div>

      {/* Main Value */}
      <div className="flex-1 flex flex-col justify-center">
        <div className="flex items-baseline gap-2 mb-1">
          <span style={{ 
            fontSize: '48px',
            fontWeight: 'var(--font-weight-bold)',
            color: 'var(--text-primary)',
            lineHeight: '1'
          }}>
            {riskLevel ? (
              <span className="flex items-center gap-2">
                <AlertTriangle className="w-10 h-10" style={{ color: riskConfig[riskLevel].text }} />
                <span style={{ fontSize: '32px' }}>{value}</span>
              </span>
            ) : (
              value
            )}
          </span>
          {riskLevel && (
            <div 
              className="px-3 py-1.5 rounded-2xl text-sm"
              style={{
                backgroundColor: riskConfig[riskLevel].bg,
                color: riskConfig[riskLevel].text,
                fontWeight: 'var(--font-weight-semibold)'
              }}
            >
              {value}
            </div>
          )}
        </div>

        {/* Subtitle / Detail */}
        {subtitle && (
          <div className="text-sm mt-1" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: 'var(--text-secondary)'
          }}>
            {subtitle}
          </div>
        )}

        {detail && (
          <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            {detail}
          </div>
        )}

        {/* Trend Indicator */}
        {trend && trendDirection && (
          <div 
            className="flex items-center gap-1 mt-2 text-sm"
            style={{
              color: trendDirection === 'up' ? 'var(--success-green)' : 'var(--danger-red)',
              fontWeight: 'var(--font-weight-medium)'
            }}
          >
            <TrendIcon className="w-4 h-4" />
            {trend}
          </div>
        )}

        {/* Live Indicator */}
        {isLive && (
          <div className="flex items-center gap-2 mt-2">
            <div 
              className="w-2 h-2 rounded-full animate-pulse"
              style={{ backgroundColor: 'var(--success-green)' }}
            />
            <span className="text-xs" style={{ 
              color: 'var(--success-green)',
              fontWeight: 'var(--font-weight-medium)'
            }}>
              Live
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

interface StatusItemProps {
  label: string;
  status: 'connected' | 'degraded' | 'disconnected';
  detail: string;
}

function StatusItem({ label, status, detail }: StatusItemProps) {
  const statusConfig = {
    connected: { color: 'var(--status-connected)', text: 'Connected' },
    degraded: { color: 'var(--status-degraded)', text: 'Degraded' },
    disconnected: { color: 'var(--status-disconnected)', text: 'Disconnected' }
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-3">
        <div 
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: config.color }}
        />
        <div>
          <p className="text-sm" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: 'var(--text-primary)'
          }}>
            {label}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {detail}
          </p>
        </div>
      </div>
      <span 
        className="text-xs px-2 py-1 rounded"
        style={{ 
          backgroundColor: `${config.color}15`,
          color: config.color,
          fontWeight: 'var(--font-weight-medium)'
        }}
      >
        {config.text}
      </span>
    </div>
  );
}