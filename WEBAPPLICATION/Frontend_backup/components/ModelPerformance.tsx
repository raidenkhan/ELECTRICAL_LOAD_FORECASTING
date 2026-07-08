import { Crown, TrendingUp, CheckCircle, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useState } from 'react';

const performanceData = [
  { date: 'Jan 5', baseline: 11.8, champion: 9.2 },
  { date: 'Jan 10', baseline: 11.5, champion: 8.5 },
  { date: 'Jan 15', baseline: 11.2, champion: 7.8 },
  { date: 'Jan 20', baseline: 11.0, champion: 7.4 },
  { date: 'Jan 25', baseline: 10.8, champion: 7.3 },
  { date: 'Jan 30', baseline: 10.5, champion: 7.1 },
  { date: 'Feb 4', baseline: 10.3, champion: 7.2 }
];

const errorHeatmapData = [
  { month: 'Jan', '00-04': 'low', '04-08': 'low', '08-12': 'medium', '12-16': 'high', '16-20': 'high', '20-24': 'medium' },
  { month: 'Feb', '00-04': 'low', '04-08': 'low', '08-12': 'low', '12-16': 'medium', '16-20': 'high', '20-24': 'low' },
  { month: 'Mar', '00-04': 'medium', '04-08': 'low', '08-12': 'low', '12-16': 'medium', '16-20': 'medium', '20-24': 'low' },
  { month: 'Apr', '00-04': 'low', '04-08': 'low', '08-12': 'medium', '12-16': 'high', '16-20': 'high', '20-24': 'medium' }
];

const featureImportance = [
  { feature: 'Lag_96 (yesterday)', contribution: 12, percentage: 100 },
  { feature: 'NY6ZA_Flow', contribution: 8, percentage: 67 },
  { feature: 'Month (Feb)', contribution: 5, percentage: 42 },
  { feature: 'Hour (14)', contribution: 4, percentage: 33 },
  { feature: 'Regime (Seasonal)', contribution: 3, percentage: 25 }
];

export function ModelPerformance() {
  return (
    <div className="space-y-6">
      {/* Model Performance Monitor */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
        }}
      >
        <h3 className="mb-6" style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Model Performance Monitor
        </h3>

        {/* Metric Cards */}
        <div className="grid grid-cols-3 gap-5 mb-6">
          <ChampionCard />
          <MetricCard
            label="Current MAE"
            value="7.2 MW"
          />
          <MetricCard
            label="vs Baseline"
            value="39% Better"
            isPositive={true}
          />
        </div>

        {/* Performance Trend */}
        <div className="mb-6">
          <h4 className="mb-4 text-sm" style={{ 
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--text-secondary)'
          }}>
            Performance Trend (Last 30 Days)
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
              <XAxis 
                dataKey="date" 
                stroke="var(--text-muted)"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="var(--text-muted)"
                style={{ fontSize: '12px' }}
                label={{ value: 'MAE (MW)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)' } }}
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
                dataKey="baseline" 
                stroke="#94A3B8" 
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Baseline"
                dot={{ fill: '#94A3B8', r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="champion" 
                stroke="var(--primary-blue)" 
                strokeWidth={2}
                name="Champion"
                dot={{ fill: 'var(--primary-blue)', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Status */}
        <div className="flex items-center gap-3 pt-4" style={{ borderTop: '1px solid var(--border-default)' }}>
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5" style={{ color: 'var(--success-green)' }} />
            <span className="text-sm" style={{ 
              fontWeight: 'var(--font-weight-medium)',
              color: 'var(--success-green)'
            }}>
              Status: Healthy
            </span>
          </div>
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>|</span>
          <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            No drift detected
          </span>
        </div>
      </div>

      {/* Error Heatmap */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
        }}
      >
        <h3 className="mb-6" style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Error Distribution: Hour × Month
        </h3>

        <ErrorHeatmap data={errorHeatmapData} />

        {/* Legend */}
        <div className="flex items-center gap-6 mt-4 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded" style={{ backgroundColor: '#DCFCE7' }} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>&lt;5 MW</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded" style={{ backgroundColor: '#FEF3C7' }} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>5-10 MW</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded" style={{ backgroundColor: '#FEE2E2' }} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>&gt;10 MW</span>
          </div>
        </div>

        {/* Insight */}
        <div 
          className="px-4 py-3 rounded-md flex items-start gap-3"
          style={{
            backgroundColor: '#EFF6FF',
            border: '1px solid var(--primary-blue)'
          }}
        >
          <span className="text-lg">💡</span>
          <div>
            <span className="text-sm" style={{ 
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--text-primary)'
            }}>
              Insight:
            </span>
            <span className="text-sm ml-1" style={{ color: 'var(--text-secondary)' }}>
              Highest errors during peak hours (12-20)
            </span>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div 
        className="rounded-lg p-6"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-default)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
        }}
      >
        <h3 className="mb-2" style={{ 
          fontSize: 'var(--text-lg)',
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-primary)'
        }}>
          Why is the forecast high?
        </h3>
        <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
          Prediction: 102 MW on Feb 4, 14:30
        </p>

        <h4 className="text-sm mb-4" style={{ 
          fontWeight: 'var(--font-weight-semibold)',
          color: 'var(--text-secondary)'
        }}>
          Top Contributing Features:
        </h4>

        <div className="space-y-3 mb-6">
          {featureImportance.map((item, index) => (
            <FeatureBar key={index} {...item} />
          ))}
        </div>

        {/* Summary */}
        <div 
          className="flex items-center justify-center gap-3 py-4 px-6 rounded-md text-sm mb-4"
          style={{
            backgroundColor: '#F8FAFC',
            border: '1px solid var(--border-default)'
          }}
        >
          <span style={{ color: 'var(--text-secondary)' }}>Base: <strong>70 MW</strong></span>
          <span style={{ color: 'var(--text-muted)' }}>+</span>
          <span style={{ color: 'var(--text-secondary)' }}>Adjustments: <strong style={{ color: 'var(--primary-blue)' }}>+32 MW</strong></span>
          <span style={{ color: 'var(--text-muted)' }}>=</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 'var(--font-weight-semibold)' }}>Final: <strong>102 MW</strong></span>
        </div>

        <button
          className="text-sm px-4 py-2 rounded-md transition-colors flex items-center gap-2"
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
  );
}

function ChampionCard() {
  return (
    <div 
      className="rounded-lg p-5 relative overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)',
        border: '1px solid var(--warning-orange)'
      }}
    >
      <div className="absolute top-3 right-3 text-2xl">👑</div>
      <p className="text-xs mb-2" style={{ 
        fontWeight: 'var(--font-weight-medium)',
        color: '#92400E'
      }}>
        Champion
      </p>
      <p className="text-xl mb-1" style={{ 
        fontWeight: 'var(--font-weight-bold)',
        color: 'var(--text-primary)'
      }}>
        LightGBM v3
      </p>
      <p className="text-sm" style={{ color: '#92400E' }}>
        Jan 15
      </p>
    </div>
  );
}

function MetricCard({ label, value, isPositive }: { label: string; value: string; isPositive?: boolean }) {
  return (
    <div 
      className="rounded-lg p-5"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-default)'
      }}
    >
      <p className="text-xs mb-2" style={{ 
        fontWeight: 'var(--font-weight-medium)',
        color: '#64748B'
      }}>
        {label}
      </p>
      <div className="flex items-center gap-2">
        <p className="text-xl" style={{ 
          fontWeight: 'var(--font-weight-bold)',
          color: 'var(--text-primary)'
        }}>
          {value}
        </p>
        {isPositive && (
          <CheckCircle className="w-5 h-5" style={{ color: 'var(--success-green)' }} />
        )}
      </div>
    </div>
  );
}

function ErrorHeatmap({ data }: { data: any[] }) {
  const timeSlots = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24'];
  
  const getColor = (level: string) => {
    switch (level) {
      case 'low': return '#DCFCE7';
      case 'medium': return '#FEF3C7';
      case 'high': return '#FEE2E2';
      default: return '#F1F5F9';
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left pb-3 pr-4" style={{ 
              fontSize: '12px',
              fontWeight: 'var(--font-weight-semibold)',
              color: '#64748B'
            }}>
              Month
            </th>
            {timeSlots.map(slot => (
              <th key={slot} className="text-center pb-3 px-2" style={{ 
                fontSize: '12px',
                fontWeight: 'var(--font-weight-semibold)',
                color: '#64748B'
              }}>
                {slot}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td className="py-2 pr-4 text-sm" style={{ 
                fontWeight: 'var(--font-weight-medium)',
                color: 'var(--text-primary)'
              }}>
                {row.month}
              </td>
              {timeSlots.map(slot => (
                <td key={slot} className="px-2 py-2">
                  <div 
                    className="w-full h-12 rounded"
                    style={{ backgroundColor: getColor(row[slot]) }}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureBar({ feature, contribution, percentage }: { feature: string; contribution: number; percentage: number }) {
  return (
    <div className="flex items-center gap-4">
      <span className="text-sm w-48 flex-shrink-0" style={{ 
        color: 'var(--text-secondary)',
        fontWeight: 'var(--font-weight-medium)'
      }}>
        {feature}
      </span>
      <div className="flex-1 relative">
        <div 
          className="h-2 rounded-full"
          style={{ backgroundColor: '#E2E8F0' }}
        >
          <div 
            className="h-2 rounded-full transition-all duration-300"
            style={{ 
              backgroundColor: 'var(--primary-blue)',
              width: `${percentage}%`
            }}
          />
        </div>
      </div>
      <span className="text-sm w-16 text-right flex-shrink-0" style={{ 
        fontWeight: 'var(--font-weight-semibold)',
        color: 'var(--primary-blue)'
      }}>
        +{contribution} MW
      </span>
    </div>
  );
}
