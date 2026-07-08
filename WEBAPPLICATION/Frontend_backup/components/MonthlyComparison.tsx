import { TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const comparisonData = [
  { day: '1', current: 88, previous: 85 },
  { day: '5', current: 92, previous: 88 },
  { day: '10', current: 98, previous: 95 },
  { day: '15', current: 105, previous: 102 },
  { day: '20', current: 102, previous: 98 },
  { day: '25', current: 95, previous: 92 },
  { day: '28', current: 90, previous: 87 }
];

export function MonthlyComparison() {
  return (
    <div 
      className="rounded-lg p-6"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
      }}
    >
      {/* Header */}
      <div className="mb-6">
        <h3 className="mb-3" style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Monthly Comparison
        </h3>
        <div className="flex items-center gap-3">
          <select 
            className="px-3 py-2 rounded-md text-sm"
            style={{
              border: '1px solid var(--border-default)',
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)'
            }}
          >
            <option>Feb 2026</option>
            <option>Jan 2026</option>
            <option>Dec 2025</option>
          </select>
          <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>vs</span>
          <select 
            className="px-3 py-2 rounded-md text-sm"
            style={{
              border: '1px solid var(--border-default)',
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)'
            }}
          >
            <option>Feb 2025</option>
            <option>Jan 2025</option>
            <option>Dec 2024</option>
          </select>
        </div>
      </div>

      {/* Comparison Cards */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div 
          className="rounded-lg p-5"
          style={{
            background: 'linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)'
          }}
        >
          <p className="text-xs mb-2" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: '#1E40AF'
          }}>
            Avg Peak
          </p>
          <p className="text-2xl mb-1" style={{ 
            fontWeight: 'var(--font-weight-bold)',
            color: 'var(--text-primary)'
          }}>
            95.2 MW
          </p>
          <div className="flex items-center gap-1 text-sm" style={{ color: 'var(--success-green)' }}>
            <TrendingUp className="w-4 h-4" />
            <span style={{ fontWeight: 'var(--font-weight-medium)' }}>+3.4% vs 2025</span>
          </div>
        </div>

        <div 
          className="rounded-lg p-5"
          style={{
            background: 'linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)'
          }}
        >
          <p className="text-xs mb-2" style={{ 
            fontWeight: 'var(--font-weight-medium)',
            color: '#1E40AF'
          }}>
            Max Peak
          </p>
          <p className="text-2xl mb-1" style={{ 
            fontWeight: 'var(--font-weight-bold)',
            color: 'var(--text-primary)'
          }}>
            108 MW
          </p>
          <div className="flex items-center gap-1 text-sm" style={{ color: 'var(--success-green)' }}>
            <TrendingUp className="w-4 h-4" />
            <span style={{ fontWeight: 'var(--font-weight-medium)' }}>+2.9% vs 2025</span>
          </div>
        </div>
      </div>

      {/* Comparison Chart */}
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={comparisonData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis 
            dataKey="day" 
            stroke="var(--text-muted)"
            style={{ fontSize: '12px' }}
            label={{ value: 'Day of Month', position: 'insideBottom', offset: -5, style: { fill: 'var(--text-secondary)' } }}
          />
          <YAxis 
            stroke="var(--text-muted)"
            style={{ fontSize: '12px' }}
            label={{ value: 'Peak Load (MW)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
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
          <Line 
            type="monotone" 
            dataKey="current" 
            stroke="var(--primary-blue)" 
            strokeWidth={2}
            name="Feb 2026"
            dot={{ fill: 'var(--primary-blue)', r: 4 }}
          />
          <Line 
            type="monotone" 
            dataKey="previous" 
            stroke="#94A3B8" 
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Feb 2025"
            dot={{ fill: '#94A3B8', r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
