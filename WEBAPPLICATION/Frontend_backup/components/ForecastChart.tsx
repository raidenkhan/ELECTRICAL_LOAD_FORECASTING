import { Settings, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useState } from 'react';

const forecastData = [
  { time: 'Now', forecast: 65, upper: 72, lower: 58 },
  { time: '6h', forecast: 75, upper: 85, lower: 68 },
  { time: '12h', forecast: 95, upper: 108, lower: 88 },
  { time: '18h', forecast: 110, upper: 120, lower: 98, peak: true },
  { time: '24h', forecast: 70, upper: 80, lower: 62 }
];

export function ForecastChart() {
  const [showP10P90, setShowP10P90] = useState(false);
  const [showConfidenceBands, setShowConfidenceBands] = useState(true);
  const [showHistorical, setShowHistorical] = useState(false);

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
          Next 24 Hours Forecast
        </h3>
        <button 
          className="flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors"
          style={{
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-default)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">Settings</span>
        </button>
      </div>

      {/* Chart Area */}
      <div className="px-6 py-6">
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={forecastData}>
            <defs>
              <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#DBEAFE" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#DBEAFE" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis 
              dataKey="time" 
              stroke="var(--text-muted)"
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="var(--text-muted)"
              style={{ fontSize: '12px' }}
              label={{ value: 'MW', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-default)',
                borderRadius: '6px',
                fontSize: '12px'
              }}
            />
            {showConfidenceBands && (
              <>
                <Area 
                  type="monotone" 
                  dataKey="upper" 
                  stroke="transparent" 
                  fill="url(#confidenceFill)"
                  name="Upper Bound"
                />
                <Area 
                  type="monotone" 
                  dataKey="lower" 
                  stroke="transparent" 
                  fill="url(#confidenceFill)"
                  name="Lower Bound"
                />
              </>
            )}
            <Line 
              type="monotone" 
              dataKey="forecast" 
              stroke="var(--primary-blue)" 
              strokeWidth={2}
              dot={(props: any) => {
                if (props.payload.peak) {
                  return (
                    <circle 
                      cx={props.cx} 
                      cy={props.cy} 
                      r={4} 
                      fill="var(--danger-red)" 
                      stroke="var(--danger-red)"
                      strokeWidth={2}
                    />
                  );
                }
                return <circle cx={props.cx} cy={props.cy} r={3} fill="var(--primary-blue)" />;
              }}
              name="Forecast"
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* Controls */}
        <div className="flex items-center gap-6 mt-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showP10P90}
              onChange={(e) => setShowP10P90(e.target.checked)}
              className="w-4 h-4 rounded cursor-pointer"
              style={{ accentColor: 'var(--primary-blue)' }}
            />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Show P10/P90
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showConfidenceBands}
              onChange={(e) => setShowConfidenceBands(e.target.checked)}
              className="w-4 h-4 rounded cursor-pointer"
              style={{ accentColor: 'var(--primary-blue)' }}
            />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Confidence Bands
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showHistorical}
              onChange={(e) => setShowHistorical(e.target.checked)}
              className="w-4 h-4 rounded cursor-pointer"
              style={{ accentColor: 'var(--primary-blue)' }}
            />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Historical
            </span>
          </label>
        </div>
      </div>

      {/* Alert Banner */}
      <div 
        className="mx-6 mb-6 px-4 py-3 rounded-md flex items-start gap-3"
        style={{
          backgroundColor: '#FEF3C7',
          borderLeft: '4px solid var(--warning-orange)'
        }}
      >
        <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--warning-orange)' }} />
        <div className="flex-1">
          <p className="text-sm" style={{ 
            fontWeight: 'var(--font-weight-semibold)',
            color: '#92400E'
          }}>
            HIGH VOLATILITY WINDOW: 14:00 - 16:00
          </p>
          <p className="text-sm mt-0.5" style={{ color: '#92400E' }}>
            Consider deploying spinning reserve
          </p>
        </div>
      </div>
    </div>
  );
}
