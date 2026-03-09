import { X, AlertTriangle, Loader2, Info, BarChart3 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { forecastService } from '@/services/forecastService';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface ExplainabilityModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ExplainabilityModal({ isOpen, onClose }: ExplainabilityModalProps) {
  const [shapData, setShapData] = useState<{ features: string[], values: number[], base_value: number } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      const fetchData = async () => {
        try {
          setIsLoading(true);
          const data = await forecastService.getShapValues();
          setShapData(data);
        } catch (err) {
          console.error('Failed to fetch SHAP:', err);
        } finally {
          setIsLoading(false);
        }
      };
      fetchData();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const totalAdjustment = shapData ? shapData.values.reduce((a, b) => a + b, 0) : 0;
  const finalForecast = shapData ? Math.round(shapData.base_value + totalAdjustment) : 1580;

  // Prepare data for BarChart
  const chartData = shapData ? shapData.features.map((f, i) => ({
    name: f.replace(/_mw|mw/gi, '').replace(/_/g, ' ').toUpperCase(),
    value: parseFloat(shapData.values[i].toFixed(1)),
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : [];

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50 p-4"
      style={{
        backgroundColor: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(8px)'
      }}
      onClick={onClose}
    >
      <div
        className="rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto flex flex-col"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid var(--border-primary)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-8 border-b" style={{ borderColor: 'var(--border-primary)' }}>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h2 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
                  Forecast Insights
                </h2>
                <div className="px-2 py-0.5 rounded bg-blue-50 text-[10px] font-bold text-blue-600 uppercase">
                  SHAP Explainability
                </div>
              </div>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {isLoading ? 'Running attribution model...' : `Target: ${finalForecast} MW (+${totalAdjustment.toFixed(1)} MW from base)`}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-gray-100 transition-colors"
              style={{ color: 'var(--text-muted)' }}
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-8 space-y-8 overflow-y-auto">

          {/* Top Drivers Chart */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-widest flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                <BarChart3 className="w-4 h-4" /> Feature Impact (MW)
              </h3>
              <div className="flex items-center gap-4 text-[10px] font-medium">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-red-500" /> Increasing Demand
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" /> Reducing Demand
                </div>
              </div>
            </div>

            <div className={`h-[300px] w-full rounded-xl border p-4 bg-gray-50/30 relative ${isLoading ? 'opacity-50' : ''}`}
              style={{ borderColor: 'var(--border-default)' }}>
              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-2" />
                  <span className="text-xs font-medium text-gray-500">Calculating...</span>
                </div>
              )}

              {!isLoading && chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                    <XAxis type="number" hide />
                    <YAxis
                      dataKey="name"
                      type="category"
                      stroke="#64748b"
                      fontSize={10}
                      width={80}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: 'transparent' }}
                      contentStyle={{ borderRadius: '8px', border: '1px solid var(--border-default)', fontSize: '12px', fontWeight: '500' }}
                      formatter={(value: number) => [`${value > 0 ? '+' : ''}${value} MW`, 'Impact']}
                    />
                    <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : !isLoading && (
                <div className="h-full flex items-center justify-center text-gray-400 text-sm italic">
                  No driver data available for this horizon.
                </div>
              )}
            </div>
          </div>

          {/* Contextual Intelligence */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border bg-blue-50/50" style={{ borderColor: '#bfdbfe' }}>
              <h4 className="text-xs font-bold text-blue-800 uppercase mb-2 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" /> Regime Analysis
              </h4>
              <p className="text-sm font-semibold text-blue-900 mb-1">Seasonal Peak Threshold</p>
              <p className="text-xs text-blue-700 leading-relaxed">
                The current load profile matches historical "Industrial Startup" patterns (6AM-9AM).
              </p>
            </div>

            <div className="p-4 rounded-xl border bg-amber-50/50" style={{ borderColor: '#fde68a' }}>
              <h4 className="text-xs font-bold text-amber-800 uppercase mb-2 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5" /> Confidence Check
              </h4>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-bold text-amber-900">MEDIUM (84%)</span>
              </div>
              <p className="text-xs text-amber-700 leading-relaxed">
                Higher than usual variance in temperature sensor inputs for this specific node.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50/50 flex items-center justify-between" style={{ borderColor: 'var(--border-primary)' }}>
          <p className="text-[10px] text-gray-400 font-medium">
            Model: LTLF-Recursive-v2.1
          </p>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-5 py-2 text-sm font-semibold rounded-lg border transition-all hover:bg-white active:scale-95"
              style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)' }}
            >
              Close
            </button>
            <button
              className="px-5 py-2 text-sm font-bold text-white rounded-lg transition-all active:scale-95 shadow-lg shadow-blue-500/20"
              style={{ backgroundColor: 'var(--primary-blue)' }}
            >
              Download Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
