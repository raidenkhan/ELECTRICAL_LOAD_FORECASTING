import { X, AlertTriangle, Loader2, Info, BarChart3 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { forecastService } from '@/services/forecastService';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface ExplainabilityModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ExplainabilityModal({ isOpen, onClose }: ExplainabilityModalProps) {
  const [decomData, setDecomData] = useState<{ peak_mw: number, peak_hour: number, mean_mw: number, components: { name: string, value: number, color: string }[] } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      const fetchData = async () => {
        try {
          setIsLoading(true);
          const data = await forecastService.getPeakDecomposition();
          setDecomData(data);
        } catch (err) {
          console.error('Failed to fetch decomposition:', err);
        } finally {
          setIsLoading(false);
        }
      };
      fetchData();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  // Prepare data for stacked comparison
  // Recharts Bar chart can handle multiple <Bar /> components for stacking
  const chartData = decomData ? [
    decomData.components.reduce((acc, curr) => ({ ...acc, [curr.name]: curr.value }), { name: 'Peak Hour Breakdown' })
  ] : [];

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
                  Structural Insights
                </h2>
                <div className="px-2 py-0.5 rounded bg-emerald-50 text-[10px] font-bold text-emerald-600 uppercase">
                  Decomposition Engine
                </div>
              </div>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {isLoading ? 'Decomposing signal...' : `Peak Demand Analysis: ${decomData?.peak_mw} MW at H${(decomData?.peak_hour ?? 0).toString().padStart(2, '0')}:00`}
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

          {/* Component Breakdown Chart */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-widest flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                <BarChart3 className="w-4 h-4" /> Component Impact (MW)
              </h3>
            </div>

            <div className={`h-[300px] w-full rounded-xl border p-4 bg-gray-50/30 relative ${isLoading ? 'opacity-50' : ''}`}
              style={{ borderColor: 'var(--border-default)' }}>
              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-2" />
                  <span className="text-xs font-medium text-gray-500">Decomposing...</span>
                </div>
              )}

              {!isLoading && decomData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 20, right: 30, left: 40, bottom: 20 }} barSize={60}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} vertical={true} stroke="#e2e8f0" />
                    <XAxis type="number" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} label={{ value: 'Demand (MW)', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#64748b' }} />
                    <YAxis dataKey="name" type="category" hide />
                    <Tooltip
                      cursor={{ fill: 'transparent' }}
                      contentStyle={{ borderRadius: '12px', border: '1px solid var(--border-default)', fontSize: '13px', fontWeight: 'bold' }}
                      formatter={(value: number, name: string) => [`${value > 0 ? '+' : ''}${value} MW`, name]}
                    />
                    {decomData.components.map((comp, i) => (
                      <Bar 
                        key={comp.name} 
                        dataKey={comp.name} 
                        stackId="peak" 
                        fill={comp.color} 
                        radius={i === 0 ? [4, 0, 0, 4] : i === decomData.components.length - 1 ? [0, 4, 4, 0] : [0, 0, 0, 0]} 
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : !isLoading && (
                <div className="h-full flex items-center justify-center text-gray-400 text-sm italic">
                  No structural data available for this horizon.
                </div>
              )}
            </div>
            
            {/* Legend for components */}
            <div className="mt-6 flex flex-wrap gap-4 justify-center">
              {decomData?.components.map((comp) => (
                <div key={comp.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: comp.color }} />
                  <span className="text-[11px] font-bold uppercase tracking-tight text-gray-600">{comp.name}</span>
                </div>
              ))}
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
            Model: DLinear+TIDE v2.0
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
