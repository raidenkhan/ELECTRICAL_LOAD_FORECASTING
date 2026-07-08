import React from 'react';
import { StatusBadge } from './StatusBadge';
import { DispatchForecastResponse } from '@/services/dispatchForecastService';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  Area, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip 
} from 'recharts';

interface WeeklyForecastProps {
  data?: DispatchForecastResponse | null;
  isLoading?: boolean;
}

// Helper to generate a simple SVG sparkline path
const generateSparkline = (pointsData?: number[]) => {
  if (!pointsData || pointsData.length === 0) {
    const points = Array.from({ length: 24 }).map((_, i) => {
      const y = 30 + Math.sin(i / 12 * Math.PI) * 15 + Math.random() * 5;
      return `${i * 10},${y}`;
    });
    return `M ${points.join(' L ')}`;
  }
  
  const max = Math.max(...pointsData);
  const min = Math.min(...pointsData);
  const range = max - min || 1;
  
  const points = pointsData.map((val, i) => {
    // Normalize to 0-40 height
    const normalized = ((val - min) / range) * 40;
    const y = 50 - normalized; // Invert Y for SVG
    const x = (i / (pointsData.length - 1)) * 230;
    return `${x},${y}`;
  });

  return `M ${points.join(' L ')}`;
};

export function WeeklyForecast({ data, isLoading }: WeeklyForecastProps) {
  // Parse single-day 24H forecast data
  const days: any[] = [];
  
  if (data && data.forecast_mw && data.forecast_mw.length > 0) {
    const peakVal = Math.max(...data.forecast_mw);
    const peakIdx = data.forecast_mw.indexOf(peakVal);
    const dateStr = data.forecast_date
      ? new Date(data.forecast_date + 'T12:00:00').toLocaleDateString([], { month: 'short', day: 'numeric' })
      : '---';

    days.push({
      name: new Date((data.forecast_date || '').split('-').join('/')).toLocaleDateString([], { weekday: 'long' }),
      date: dateStr,
      peak: Math.round(peakVal),
      time: `H${(peakIdx + 1).toString().padStart(2, '0')}:00`,
      holiday: false,
      critical: peakVal > 200,
      points: data.forecast_mw
    });
  }

  // Fallback while loading or if parsing failed
  const displayDays = days.length > 0 ? days : Array.from({ length: 7 }).map((_, i) => ({
    name: 'Loading',
    date: '---',
    peak: 0,
    time: '--:--',
    holiday: false,
    critical: false,
    points: []
  }));

  const maxPeakWeek = Math.max(...displayDays.map(d => d.peak));
  const isMacroHorizon = displayDays.length > 7;

  return (
    <div className="space-y-6">
      
      {isMacroHorizon ? (
        <div className="glass-panel p-6 h-[400px] w-full border border-[var(--divider)]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={displayDays} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="macroFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--brand-blue-vibrant)" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="var(--brand-blue-vibrant)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="0" stroke="var(--chart-grid)" vertical={false} strokeWidth={0.5} />
                <XAxis 
                  dataKey="date" 
                  stroke="var(--text-muted)" 
                  fontSize={10} 
                  fontFamily="JetBrains Mono"
                  tickMargin={10}
                  axisLine={false}
                  tickLine={false}
                  minTickGap={30}
                />
                <YAxis 
                  stroke="var(--text-muted)" 
                  fontSize={10} 
                  fontFamily="JetBrains Mono"
                  axisLine={false}
                  tickLine={false}
                  domain={['auto', 'auto']}
                  tickFormatter={v => `${v} MW`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--bg-card)', 
                    borderColor: 'var(--border-card)', 
                    borderRadius: '6px',
                    fontFamily: 'JetBrains Mono',
                    fontSize: '11px'
                  }}
                  cursor={{ stroke: 'var(--brand-teal)', strokeWidth: 1, strokeDasharray: '4 4' }}
                />
                <Area
                  type="monotone"
                  dataKey="peak"
                  fill="url(#macroFill)"
                  stroke="none"
                />
                <Line
                  type="monotone"
                  dataKey="peak"
                  stroke="var(--brand-blue-vibrant)"
                  strokeWidth={2}
                  dot={displayDays.length < 40 ? { fill: 'var(--brand-blue-vibrant)', r: 2 } : false}
                  activeDot={{ r: 5, fill: 'var(--brand-gold)', stroke: 'var(--bg-card)', strokeWidth: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
        </div>
      ) : (
      <div className="grid grid-cols-1 md:grid-cols-4 xl:grid-cols-7 gap-4">
        {displayDays.map((day, idx) => (
          <div 
            key={`${day.name}-${idx}`} 
            className={`glass-panel p-5 flex flex-col gap-5 relative group transition-all duration-300 border-t-2
              ${day.critical ? 'border-t-[var(--status-crimson)]' : 'border-t-transparent hover:border-t-[var(--brand-blue)]'}
            `}
          >
            {/* Header: Day & Date */}
            <div className="flex justify-between items-start">
              <div className="flex flex-col">
                <span className="text-[11px] font-black text-[var(--text-primary)] uppercase tracking-[0.15em]">{day.name}</span>
                <span className="text-[10px] font-bold text-[var(--text-muted)] font-mono uppercase">{day.date}</span>
              </div>
              {day.critical && (
                <div className="px-1.5 py-0.5 bg-[var(--status-crimson)]/10 rounded-sm">
                  <span className="text-[8px] font-black text-[var(--status-crimson)] uppercase tracking-tighter">PEAK</span>
                </div>
              )}
            </div>

            {/* Sparkline Visual */}
            <div className="h-12 w-full opacity-60 group-hover:opacity-100 transition-opacity">
              {isLoading ? (
                <div className="w-full h-full animate-pulse bg-[var(--divider)]/20 rounded-sm" />
              ) : (
                <svg viewBox="0 0 230 60" className="w-full h-full overflow-visible">
                  <path 
                    d={generateSparkline(day.points)} 
                    fill="none" 
                    stroke={day.critical ? 'var(--status-crimson)' : 'var(--brand-indigo)'} 
                    strokeWidth="3" 
                    strokeLinecap="round" 
                  />
                </svg>
              )}
            </div>

            {/* Metrics: Peak & Time */}
            <div className="flex flex-col gap-1 mt-auto">
              <div className="flex items-baseline gap-1.5">
                <span className="metric-num text-2xl font-bold text-[var(--text-primary)] leading-none">
                  {isLoading ? "..." : day.peak.toLocaleString()}
                </span>
                <span className="text-[10px] font-black text-[var(--text-muted)] uppercase">MW</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-[9px] font-bold text-[var(--text-secondary)] uppercase tracking-widest">
                  Peak at {isLoading ? "--:--" : day.time}
                </span>
              </div>
            </div>

            {day.holiday && (
              <div className="pt-3 border-t border-[var(--divider)]/30">
                <StatusBadge status="holiday" label="Public Holiday" />
              </div>
            )}
          </div>
        ))}
      </div>
      )}

      {/* Professional Alert Banner */}
      {!isLoading && maxPeakWeek > 145 && (
        <div className="p-4 bg-[var(--status-crimson)]/5 border border-[var(--status-crimson)]/20 rounded-sm flex items-start gap-4 glass-panel">
          <div className="w-10 h-10 flex-shrink-0 rounded-sm bg-[var(--status-crimson)]/10 flex items-center justify-center">
            <span className="text-xl font-black text-[var(--status-crimson)]">!</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] font-black text-[var(--status-crimson)] uppercase tracking-[0.2em]">Operational Risk Alert</span>
            <p className="text-[13px] font-medium text-[var(--text-primary)] leading-snug">
              Load projection exceeds typical substation maximums (145 MW). 
              Automated recommendation: Review transformer cooling systems and busbar alignments.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

