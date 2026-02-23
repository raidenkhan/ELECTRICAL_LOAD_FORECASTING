'use client';

import { Settings, AlertTriangle, Info, X } from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, Line } from 'recharts';
import { useState, useRef, useEffect } from 'react';

interface ForecastDataPoint {
  time: string;
  forecast: number;
  upper?: number;
  lower?: number;
  peak?: boolean;
}

interface ForecastChartProps {
  data?: ForecastDataPoint[];
  isLoading?: boolean;
}

const mockData: ForecastDataPoint[] = [
  { time: 'Now', forecast: 1247, upper: 1320, lower: 1180 },
  { time: '6h', forecast: 1350, upper: 1450, lower: 1280 },
  { time: '12h', forecast: 1520, upper: 1640, lower: 1440 },
  { time: '18h', forecast: 1580, upper: 1680, lower: 1480, peak: true },
  { time: '24h', forecast: 1180, upper: 1280, lower: 1100 }
];

export function ForecastChart({ data = [], isLoading = false }: ForecastChartProps) {
  const [showP10P90, setShowP10P90] = useState(false);
  const [showConfidenceBands, setShowConfidenceBands] = useState(true);
  const [showHistorical, setShowHistorical] = useState(false);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const configPanelRef = useRef<HTMLDivElement>(null);

  // Close config panel on outside click
  useEffect(() => {
    if (!showConfigPanel) return;
    function handleClickOutside(e: MouseEvent) {
      if (configPanelRef.current && !configPanelRef.current.contains(e.target as Node)) {
        setShowConfigPanel(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showConfigPanel]);

  // Use passed data if it exists and has points, otherwise fallback to mock during development
  // but if explicitly empty from parent, we might want to show "No Data"
  const displayData = data && data.length > 0 ? data : (isLoading ? [] : mockData);

  return (
    <div
      className="border glass-morphism"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border-primary)',
        backdropFilter: 'var(--glass-blur)',
        boxShadow: 'var(--glass-shadow)'
      }}
    >

      {/* Header */}
      <div
        className="flex items-center justify-between px-8 py-6 border-b"
        style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}
      >

        <div className="flex items-center gap-4">
          <div className="w-2 h-5" style={{ backgroundColor: 'var(--lime-primary)' }} />
          <h3 className="text-sm font-bold tracking-[0.3em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
            Load Forecast Grid
          </h3>
        </div>



        <div className="flex items-center gap-2">
          {isLoading ? (
            <div className="flex items-center gap-2 px-3 py-1.5 border" style={{ borderColor: 'var(--border-primary)' }}>
              <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--status-info)' }} />
              <span className="text-xs font-semibold" style={{ color: 'var(--status-info)' }}>
                LOADING
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 border" style={{ borderColor: 'var(--border-primary)' }}>
              <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--status-ok)' }} />
              <span className="text-xs font-semibold" style={{ color: 'var(--status-ok)' }}>
                LIVE
              </span>
            </div>
          )}

          {/* Configure button + settings panel */}
          <div className="relative" ref={configPanelRef}>
            <button
              className="flex items-center gap-2 px-3 py-1.5 border transition-colors hover:bg-white/5"
              style={{
                color: showConfigPanel ? 'var(--lime-primary)' : 'var(--text-secondary)',
                borderColor: showConfigPanel ? 'var(--lime-primary)' : 'var(--border-primary)',
                fontFamily: 'var(--font-geist-mono)',
                fontSize: '11px',
                fontWeight: 'bold',
                textTransform: 'uppercase'
              }}
              onClick={() => setShowConfigPanel(prev => !prev)}
            >
              <Settings className="w-3.5 h-3.5" />
              <span>Configure</span>
            </button>

            {/* Config Panel Dropdown */}
            {showConfigPanel && (
              <div
                className="absolute top-full right-0 mt-2 w-64 border glass-morphism z-50"
                style={{
                  backgroundColor: 'var(--bg-surface)',
                  borderColor: 'var(--border-primary)',
                  boxShadow: 'var(--glass-shadow)'
                }}
              >
                {/* Panel Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border-primary)' }}>
                  <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
                    Chart Settings
                  </span>
                  <button onClick={() => setShowConfigPanel(false)} className="hover:opacity-60 transition-opacity">
                    <X className="w-3.5 h-3.5" style={{ color: 'var(--text-muted)' }} />
                  </button>
                </div>

                {/* Toggles */}
                <div className="py-2">
                  {[
                    { label: 'Confidence Bands', desc: 'Show ±10% shaded region', state: showConfidenceBands, setState: setShowConfidenceBands },
                    { label: 'P10/P90 Bounds', desc: 'Show probabilistic bounds', state: showP10P90, setState: setShowP10P90 },
                    { label: 'Historical Overlay', desc: 'Overlay past actuals', state: showHistorical, setState: setShowHistorical },
                  ].map(({ label, desc, state, setState }) => (
                    <button
                      key={label}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors text-left"
                      onClick={() => setState((p: boolean) => !p)}
                    >
                      <div>
                        <p className="text-[11px] font-bold uppercase tracking-wide" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>{label}</p>
                        <p className="text-[9px] uppercase opacity-40 mt-0.5" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>{desc}</p>
                      </div>
                      {/* Toggle pill */}
                      <div
                        className="relative w-8 h-4 flex-shrink-0 transition-colors duration-200"
                        style={{ backgroundColor: state ? 'var(--lime-primary)' : 'var(--border-primary)' }}
                      >
                        <div
                          className="absolute top-0.5 w-3 h-3 transition-transform duration-200"
                          style={{
                            backgroundColor: state ? '#000' : 'var(--text-muted)',
                            transform: state ? 'translateX(18px)' : 'translateX(2px)'
                          }}
                        />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

        </div>
      </div>

      {/* Chart Area */}
      <div className="px-10 py-10" style={{ opacity: isLoading ? 0.5 : 1 }}>

        {displayData.length === 0 ? (
          <div className="h-[320px] flex flex-col items-center justify-center border-2 border-dashed border-gray-100 rounded-lg">
            <Info className="w-12 h-12 text-gray-300 mb-4" />
            <p className="text-sm font-medium text-gray-500">No forecast data available</p>
            <p className="text-xs text-gray-400 mt-1">Check system status or connectivity</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={displayData}>
              <defs>
                <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--lime-primary)" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="var(--lime-primary)" stopOpacity={0.05} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" strokeOpacity={0.1} vertical={false} />


              <XAxis
                dataKey="time"
                stroke="var(--text-tertiary)"
                style={{ fontSize: '10px', fontFamily: 'var(--font-geist-mono)' }}
                tickLine={false}
                axisLine={{ stroke: '#27272A' }}
                interval={Math.floor(displayData.length / 6)}
              />
              <YAxis
                stroke="var(--text-tertiary)"
                style={{ fontSize: '11px', fontFamily: 'var(--font-geist-mono)' }}
                tickLine={false}
                axisLine={{ stroke: '#27272A' }}
                domain={['auto', 'auto']}
                label={{
                  value: 'LOAD (MW)',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fill: 'var(--text-muted)', fontSize: '10px', fontWeight: 600, fontFamily: 'var(--font-geist-mono)' }
                }}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-surface)',
                  border: '1px solid var(--border-primary)',
                  borderRadius: '0px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-geist-mono)',
                  boxShadow: 'var(--glass-shadow)',
                  padding: '8px 12px'
                }}

                labelStyle={{ color: 'var(--text-primary)', fontWeight: 700, marginBottom: '4px' }}
                itemStyle={{ color: 'var(--lime-primary)' }}
              />

              {showConfidenceBands && (
                <Area
                  type="monotone"
                  dataKey={(d) => [d.lower || d.forecast * 0.9, d.upper || d.forecast * 1.1]}
                  stroke="transparent"
                  fill="url(#confidenceFill)"
                  name="Confidence Range"
                  isAnimationActive={false}
                />
              )}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="var(--lime-primary)"
                strokeWidth={3}
                style={{ filter: 'drop-shadow(0 0 8px rgba(191,255,0,0.4))' }}

                dot={(props: any) => {
                  if (!props.payload.peak) return <path key={`dot-hidden-${props.index}`} d="" />;
                  return (
                    <g key={`dot-peak-${props.index}`}>
                      <circle
                        cx={props.cx}
                        cy={props.cy}
                        r={6}
                        fill="var(--bg-page)"
                        stroke="var(--lime-primary)"
                        strokeWidth={2}
                      />
                      <circle
                        cx={props.cx}
                        cy={props.cy}
                        r={3}
                        fill="var(--lime-primary)"
                      />
                    </g>
                  );
                }}
                name="Forecast"
              />

            </ComposedChart>
          </ResponsiveContainer>
        )}

        {/* Controls */}
        <div className="flex items-center justify-between mt-6 pt-6 border-t" style={{ borderColor: 'var(--border-primary)', opacity: 0.5 }}>


          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showConfidenceBands}
                onChange={(e) => setShowConfidenceBands(e.target.checked)}
                className="w-4 h-4 cursor-pointer"
              />
              <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Confidence Bands
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showP10P90}
                onChange={(e) => setShowP10P90(e.target.checked)}
                className="w-4 h-4 cursor-pointer"
              />
              <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                P10/P90 Bounds
              </span>
            </label>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5" style={{ backgroundColor: 'var(--lime-primary)' }} />
              <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
                Forecast
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3" style={{ backgroundColor: 'var(--lime-primary)', opacity: 0.15 }} />
              <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
                Confidence
              </span>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
