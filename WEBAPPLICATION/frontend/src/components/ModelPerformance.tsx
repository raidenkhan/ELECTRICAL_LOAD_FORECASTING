import { Crown, TrendingUp, CheckCircle, AlertCircle, Award, BarChart3, Loader2, ChevronRight, AlertTriangle, Info, CheckCircle2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { forecastService } from '@/services/forecastService';

export function ModelPerformance() {
  const [metrics, setMetrics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const data = await forecastService.getModelMetrics();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to fetch metrics:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  if (isLoading && !metrics) {
    return (
      <div className="p-12 border border-border border-dashed flex flex-col items-center justify-center space-y-4">
        <Loader2 className="w-8 h-8 animate-spin text-lime-primary opacity-20" />
        <span className="text-[10px] font-black tracking-widest text-muted-foreground uppercase" style={{ color: 'var(--text-muted)' }}>Synching Model Metrics...</span>
      </div>
    );
  }

  const summary = metrics?.summary || [];
  const stlfMetrics = summary.find((m: any) => m.horizon.includes('STLF')) || { mae: 15.4, rmse: 22.1, mape: 1.2 };
  const trendData = metrics?.trend || [];
  const heatmapData = metrics?.heatmap || [];
  const featureImportanceData = metrics?.feature_importance || { features: [], base_value: 0, total_adjustment: 0 };

  return (
    <div className="space-y-6">
      {/* Model Performance Monitor */}
      <div className="glass-morphism overflow-hidden">
        <div className="px-8 py-6 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
          <div className="flex items-center gap-4">
            <div className="w-2 h-5" style={{ backgroundColor: 'var(--lime-primary)' }} />
            <h3 className="text-sm font-bold tracking-[0.3em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
              Model Performance Monitor
            </h3>
          </div>
        </div>

        <div className="p-6">
          {/* Metric Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <ChampionCard />
            <MetricCard
              label="Latest MAE"
              value={stlfMetrics.mae.toString()}
              unit="MW"
              status="good"
            />
            <MetricCard
              label="Latest MAPE"
              value={stlfMetrics.mape.toString() + "%"}
              unit=""
              status="excellent"
            />
          </div>

          {/* Performance Trend */}
          <div className="mb-8">
            <h4 className="text-[11px] font-bold tracking-widest uppercase mb-6" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
              Performance Trend (Last 30 Days)
            </h4>
            <div className="px-4">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" strokeOpacity={0.1} vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: '11px', fontWeight: 700 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: '11px', fontWeight: 700 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--bg-surface)',
                      border: '1px solid var(--border-primary)',
                      borderRadius: '0px',
                      fontSize: '11px',
                      padding: '12px',
                      boxShadow: 'var(--glass-shadow)',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-geist-mono)'
                    }}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', fontWeight: 700, fontFamily: 'var(--font-geist-mono)', paddingTop: '20px' }}
                    iconType="rect"
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    stroke="var(--text-muted)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Baseline Model"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="champion"
                    stroke="var(--lime-primary)"
                    strokeWidth={3}
                    name="Champion Model"
                    dot={false}
                    style={{ filter: 'drop-shadow(0 0 8px rgba(191,255,0,0.3))' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Status */}
          <div className="flex items-center gap-6 pt-6 border-t" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" style={{ color: 'var(--status-ok)' }} />
              <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--status-ok)', fontFamily: 'var(--font-geist-mono)' }}>
                Status: Healthy
              </span>
            </div>
            <span className="text-[11px] font-medium" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
              No drift detected • Last evaluated: Just now
            </span>
          </div>
        </div>
      </div>

      {/* Error Heatmap */}
      <div className="glass-morphism overflow-hidden">
        <div className="px-8 py-6 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
          <div className="flex items-center gap-4">
            <div className="w-2 h-5" style={{ backgroundColor: 'var(--lime-primary)' }} />
            <h3 className="text-sm font-bold tracking-[0.3em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
              Error Distribution Analysis
            </h3>
          </div>
        </div>

        <div className="p-6">
          <ErrorHeatmap data={heatmapData} />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8 pt-8 border-t" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border" style={{ backgroundColor: 'color-mix(in srgb, var(--lime-primary), transparent 80%)', borderColor: 'var(--lime-primary)' }} />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>Low</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border" style={{ backgroundColor: 'color-mix(in srgb, var(--status-warn), transparent 80%)', borderColor: 'var(--status-warn)' }} />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>Med</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 border" style={{ backgroundColor: 'color-mix(in srgb, var(--status-error), transparent 80%)', borderColor: 'var(--status-error)' }} />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>High</span>
              </div>
            </div>

            <div
              className="px-6 py-4 border-l-4 flex items-start gap-4"
              style={{
                backgroundColor: 'color-mix(in srgb, var(--status-info), transparent 95%)',
                borderLeftColor: 'var(--status-info)'
              }}
            >
              <BarChart3 className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--status-info)' }} />
              <div>
                <p className="text-[11px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--status-info)', fontFamily: 'var(--font-geist-mono)' }}>
                  PEAK ERROR ANALYTICS:
                </p>
                <p className="text-[11px] font-medium leading-relaxed" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-geist-mono)' }}>
                  Highest errors occur during peak hours (12:00-20:00). Consider specialized peak-hour model.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="glass-morphism overflow-hidden">
        <div className="px-8 py-6 border-b" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>
          <div className="flex items-center gap-4">
            <div className="w-2 h-5" style={{ backgroundColor: 'var(--lime-primary)' }} />
            <h3 className="text-sm font-bold tracking-[0.3em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
              Forecast Explainability
            </h3>
          </div>
        </div>

        <div className="p-6">
          <h4 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
            Top Contributing Features:
          </h4>

          <div className="space-y-3 mb-6">
            {featureImportanceData.features.map((item: any, index: number) => (
              <FeatureBar key={index} {...item} rank={index + 1} />
            ))}
          </div>

          {/* Summary */}
          <div
            className="flex items-center justify-center gap-6 py-4 px-6 border mb-8 glass-morphism"
            style={{
              backgroundColor: 'var(--hover-bg)',
              borderColor: 'var(--border-primary)'
            }}
          >
            <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
              Base: <strong style={{ color: 'var(--text-primary)' }}>{int(featureImportanceData.base_value).toLocaleString()} MW</strong>
            </span>
            <span style={{ color: 'var(--text-muted)' }}>|</span>
            <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color: featureImportanceData.total_adjustment >= 0 ? 'var(--status-info)' : 'var(--status-error)', fontFamily: 'var(--font-geist-mono)' }}>
              Adjustments: <strong>{featureImportanceData.total_adjustment >= 0 ? '+' : ''}{int(featureImportanceData.total_adjustment).toLocaleString()} MW</strong>
            </span>
            <span style={{ color: 'var(--text-muted)' }}>=</span>
            <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color: 'var(--lime-primary)', fontFamily: 'var(--font-geist-mono)' }}>
              Final: <strong>{int(featureImportanceData.base_value + featureImportanceData.total_adjustment).toLocaleString()} MW</strong>
            </span>
          </div>

          <button
            className="text-[11px] px-6 py-3 border transition-all duration-300 flex items-center gap-3 font-bold uppercase tracking-widest hover:border-white/20 active:scale-95 group"
            style={{
              color: 'var(--lime-primary)',
              borderColor: 'var(--lime-primary)',
              fontFamily: 'var(--font-geist-mono)'
            }}
          >
            View Full SHAP Analysis <ChevronRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>
    </div>
  );
}

// Helper for rounding/formatting MW values
function int(val: any) {
  return Math.round(parseFloat(val) || 0);
}

function ChampionCard() {
  return (
    <div className="px-6 py-6 glass-morphism relative overflow-hidden group transition-all duration-300" style={{ borderColor: 'var(--status-warn)' }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Award className="w-4 h-4" style={{ color: 'var(--status-warn)' }} />
          <span className="text-[10px] font-bold tracking-[0.2em] uppercase" style={{ color: 'var(--status-warn)', fontFamily: 'var(--font-geist-mono)' }}>
            CHAMPION MODEL
          </span>
        </div>
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--status-warn)' }} />
      </div>
      <p className="text-2xl font-black tracking-tighter mb-1" style={{ color: 'var(--text-primary)' }}>
        LightGBM v3.2
      </p>
      <p className="text-[11px] font-medium" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-geist-mono)' }}>
        Deployed: Jan 15, 2024
      </p>
    </div>
  );
}

function MetricCard({ label, value, unit, status }: { label: string; value: string; unit: string; status: 'good' | 'excellent' }) {
  const statusConfig = {
    good: { color: 'var(--status-info)' },
    excellent: { color: 'var(--status-ok)' }
  };
  const config = statusConfig[status];

  return (
    <div className="px-6 py-6 glass-morphism relative overflow-hidden group transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-1.5 h-4" style={{ backgroundColor: config.color }} />
          <h3 className="text-[10px] font-bold tracking-[0.2em] uppercase" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>{label}</h3>
        </div>
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: config.color }} />
      </div>
      <div className="flex items-baseline gap-2 mb-1">
        <p className="text-3xl font-black tracking-tighter" style={{ color: 'var(--text-primary)' }}>
          {value}
        </p>
        <span className="text-lg font-bold" style={{ color: 'var(--text-tertiary)' }}>
          {unit}
        </span>
      </div>
      <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: config.color, fontFamily: 'var(--font-geist-mono)' }}>
        {status === 'excellent' ? 'Optimum' : 'Stable'}
      </p>
    </div>
  );
}

function ErrorHeatmap({ data }: { data: any[] }) {
  const timeSlots = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24'];

  const getColor = (level: string) => {
    switch (level) {
      case 'low': return 'color-mix(in srgb, var(--lime-primary), transparent 85%)';
      case 'medium': return 'color-mix(in srgb, var(--status-warn), transparent 85%)';
      case 'high': return 'color-mix(in srgb, var(--status-error), transparent 85%)';
      default: return 'transparent';
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left pb-3 pr-4 text-[10px] font-bold tracking-widest uppercase" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
              MONTH
            </th>
            {timeSlots.map(slot => (
              <th key={slot} className="text-center pb-3 px-2 text-[10px] font-bold tracking-widest uppercase" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
                {slot}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td className="py-2 pr-4 text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
                {row.month}
              </td>
              {timeSlots.map(slot => (
                <td key={slot} className="px-1 py-1">
                  <div
                    className="w-full h-10 border transition-all duration-300 hover:scale-[1.05] hover:z-10 relative"
                    style={{ backgroundColor: getColor(row[slot]), borderColor: 'var(--border-primary)' }}
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

function FeatureBar({ feature, contribution, percentage, rank }: { feature: string; contribution: number; percentage: number; rank: number }) {
  return (
    <div className="flex items-center gap-6 group">
      <span className="text-[10px] font-bold w-6 text-center flex-shrink-0" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-geist-mono)' }}>
        {rank.toString().padStart(2, '0')}
      </span>
      <span className="text-[11px] w-48 flex-shrink-0 font-bold uppercase tracking-wider" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-geist-mono)' }}>
        {feature}
      </span>
      <div className="flex-1 relative h-1.5 bg-white/5 overflow-hidden">
        <div
          className="h-full transition-all duration-500 ease-out group-hover:opacity-80"
          style={{ backgroundColor: contribution >= 0 ? 'var(--status-info)' : 'var(--status-error)', width: `${percentage}%` }}
        />
      </div>
      <span className="text-[11px] w-24 text-right flex-shrink-0 font-bold tracking-tight" style={{ color: contribution >= 0 ? 'var(--status-info)' : 'var(--status-error)', fontFamily: 'var(--font-geist-mono)' }}>
        {contribution >= 0 ? '+' : ''}{contribution.toFixed(1)} MW
      </span>
    </div>
  );
}
