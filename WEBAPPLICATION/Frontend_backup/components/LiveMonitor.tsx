import { Activity, Zap, TrendingUp, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from 'react';

// Simulate real-time data updates
const generateRealtimeData = () => {
  const now = new Date();
  const data = [];
  for (let i = 60; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 1000);
    const baseLoad = 1500;
    const variation = Math.sin(i / 10) * 100 + Math.random() * 50;
    data.push({
      time: time.toLocaleTimeString(),
      load: Math.round(baseLoad + variation),
      forecast: Math.round(baseLoad + variation + (Math.random() - 0.5) * 30)
    });
  }
  return data;
};

export function LiveMonitor() {
  const [realtimeData, setRealtimeData] = useState(generateRealtimeData());
  const [currentLoad, setCurrentLoad] = useState(1542);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setRealtimeData(generateRealtimeData());
      setCurrentLoad(prev => prev + (Math.random() - 0.5) * 20);
      setLastUpdate(new Date());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const currentRegime = 0; // 0: Standard, 1: Transition, 2: Seasonal High
  const regimeLabels = ['Standard Operation', 'Transition/Chaos', 'Seasonal High'];
  const regimeColors = ['var(--regime-0)', 'var(--regime-1)', 'var(--regime-2)'];

  return (
    <div className="space-y-6">
      {/* Live Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <LiveMetricCard
          title="Live Load"
          value={`${Math.round(currentLoad)} MW`}
          subtitle="Real-time measurement"
          icon={<Zap className="w-6 h-6" />}
          pulse={true}
        />
        <LiveMetricCard
          title="Current Regime"
          value={regimeLabels[currentRegime]}
          subtitle="Operating state"
          icon={<Activity className="w-6 h-6" />}
          color={regimeColors[currentRegime]}
        />
        <LiveMetricCard
          title="Next Forecast"
          value="14:30"
          subtitle="In 8 minutes"
          icon={<Clock className="w-6 h-6" />}
        />
      </div>

      {/* Real-time Chart */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)'
        }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 
              className="text-lg mb-1"
              style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--text-primary)'
              }}
            >
              Real-Time Load Monitor
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Last 60 seconds • Updated {lastUpdate.toLocaleTimeString()}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-2 h-2 rounded-full animate-pulse"
              style={{ backgroundColor: 'var(--success-green)' }}
            />
            <span className="text-sm" style={{ color: 'var(--success-green)' }}>
              Live
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={realtimeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
            <XAxis 
              dataKey="time" 
              stroke="var(--text-muted)"
              style={{ fontSize: '12px' }}
              interval="preserveStartEnd"
              tickCount={6}
            />
            <YAxis 
              stroke="var(--text-muted)"
              style={{ fontSize: '12px' }}
              domain={['dataMin - 50', 'dataMax + 50']}
              label={{ value: 'Load (MW)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                borderRadius: '6px',
                fontSize: '12px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="load" 
              stroke="var(--success-green)" 
              strokeWidth={2}
              dot={false}
              name="Actual Load"
              isAnimationActive={false}
            />
            <Line 
              type="monotone" 
              dataKey="forecast" 
              stroke="var(--primary-blue)" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Forecasted"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <PerformanceMetric
          label="Error (MAPE)"
          value="3.2%"
          status="good"
        />
        <PerformanceMetric
          label="MAE"
          value="42.5 MW"
          status="good"
        />
        <PerformanceMetric
          label="RMSE"
          value="58.3 MW"
          status="good"
        />
        <PerformanceMetric
          label="Confidence"
          value="96.8%"
          status="excellent"
        />
      </div>

      {/* Recent Predictions Table */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)'
        }}
      >
        <h3 
          className="text-lg mb-4"
          style={{ 
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-primary)'
          }}
        >
          Recent Predictions
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-default)' }}>
                <th className="text-left py-3 px-4 text-sm" style={{ 
                  fontWeight: 'var(--font-weight-semibold)',
                  color: 'var(--text-secondary)'
                }}>
                  Time
                </th>
                <th className="text-left py-3 px-4 text-sm" style={{ 
                  fontWeight: 'var(--font-weight-semibold)',
                  color: 'var(--text-secondary)'
                }}>
                  Actual
                </th>
                <th className="text-left py-3 px-4 text-sm" style={{ 
                  fontWeight: 'var(--font-weight-semibold)',
                  color: 'var(--text-secondary)'
                }}>
                  Forecast
                </th>
                <th className="text-left py-3 px-4 text-sm" style={{ 
                  fontWeight: 'var(--font-weight-semibold)',
                  color: 'var(--text-secondary)'
                }}>
                  Error
                </th>
                <th className="text-left py-3 px-4 text-sm" style={{ 
                  fontWeight: 'var(--font-weight-semibold)',
                  color: 'var(--text-secondary)'
                }}>
                  Regime
                </th>
              </tr>
            </thead>
            <tbody>
              {[
                { time: '14:00', actual: 1542, forecast: 1558, regime: 0 },
                { time: '13:30', actual: 1521, forecast: 1510, regime: 0 },
                { time: '13:00', actual: 1498, forecast: 1485, regime: 0 },
                { time: '12:30', actual: 1654, forecast: 1670, regime: 2 },
                { time: '12:00', actual: 1680, forecast: 1665, regime: 2 }
              ].map((row, index) => {
                const error = ((Math.abs(row.actual - row.forecast) / row.actual) * 100).toFixed(1);
                return (
                  <tr 
                    key={index}
                    style={{ borderBottom: '1px solid var(--border-subtle)' }}
                  >
                    <td className="py-3 px-4 text-sm" style={{ color: 'var(--text-primary)' }}>
                      {row.time}
                    </td>
                    <td className="py-3 px-4 text-sm" style={{ 
                      fontWeight: 'var(--font-weight-medium)',
                      color: 'var(--text-primary)'
                    }}>
                      {row.actual} MW
                    </td>
                    <td className="py-3 px-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {row.forecast} MW
                    </td>
                    <td className="py-3 px-4">
                      <span 
                        className="text-xs px-2 py-1 rounded"
                        style={{
                          backgroundColor: parseFloat(error) < 2 ? 'rgba(5, 150, 105, 0.1)' : 
                                          parseFloat(error) < 5 ? 'rgba(245, 158, 11, 0.1)' : 
                                          'rgba(220, 38, 38, 0.1)',
                          color: parseFloat(error) < 2 ? 'var(--success-green)' : 
                                parseFloat(error) < 5 ? 'var(--warning-orange)' : 
                                'var(--danger-red)',
                          fontWeight: 'var(--font-weight-medium)'
                        }}
                      >
                        {error}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span 
                        className="text-xs px-2 py-1 rounded"
                        style={{
                          backgroundColor: `${regimeColors[row.regime]}15`,
                          color: regimeColors[row.regime],
                          fontWeight: 'var(--font-weight-medium)'
                        }}
                      >
                        {regimeLabels[row.regime]}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

interface LiveMetricCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  pulse?: boolean;
  color?: string;
}

function LiveMetricCard({ title, value, subtitle, icon, pulse, color }: LiveMetricCardProps) {
  return (
    <div 
      className="rounded-lg p-6"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)'
      }}
    >
      <div className="flex items-start justify-between mb-4">
        <div 
          className={`w-12 h-12 rounded-lg flex items-center justify-center ${pulse ? 'animate-pulse' : ''}`}
          style={{ backgroundColor: color ? `${color}15` : 'rgba(37, 99, 235, 0.1)' }}
        >
          <div style={{ color: color || 'var(--primary-blue)' }}>
            {icon}
          </div>
        </div>
        {pulse && (
          <div 
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: 'var(--success-green)' }}
          />
        )}
      </div>
      <h3 className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
        {title}
      </h3>
      <p 
        className="text-2xl mb-1"
        style={{ 
          fontWeight: 'var(--font-weight-bold)',
          color: 'var(--text-primary)'
        }}
      >
        {value}
      </p>
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {subtitle}
      </p>
    </div>
  );
}

interface PerformanceMetricProps {
  label: string;
  value: string;
  status: 'excellent' | 'good' | 'warning';
}

function PerformanceMetric({ label, value, status }: PerformanceMetricProps) {
  const statusColor = status === 'excellent' ? 'var(--success-green)' : 
                      status === 'good' ? 'var(--primary-blue)' : 
                      'var(--warning-orange)';

  return (
    <div 
      className="rounded-lg p-4 border-l-4"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: statusColor
      }}
    >
      <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
        {label}
      </p>
      <p 
        className="text-xl"
        style={{ 
          fontWeight: 'var(--font-weight-bold)',
          color: statusColor
        }}
      >
        {value}
      </p>
    </div>
  );
}
