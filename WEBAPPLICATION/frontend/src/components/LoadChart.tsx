import React from 'react';
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Area,
  ComposedChart,
  ReferenceLine,
  Line,
  AreaChart
} from 'recharts';
import { Info } from 'lucide-react';

interface DataPoint {
  time: string;
  actual?: number | null;
  projected?: number | null;
  simday?: number | null;
  baseline?: number | null;
  lowerBound?: number | null;
  upperBound?: number | null;
  
  // Power Balance Stack (Task 1)
  domestic?: number;
  industrial?: number;
  exports?: number;
  
  drivers?: { name: string; impact: number }[];
}

interface LoadChartProps {
  data: DataPoint[];
  showUncertainty?: boolean;
  liveMarkerTime?: string;
  height?: number;
  viewType?: 'default' | 'stacked' | 'strategic';
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as DataPoint;
    const isProjected = data.projected !== null;
    
    return (
      <div className="glass-panel p-3 bg-[var(--bg-card)]/95 border-[var(--border-card)] shadow-2xl min-w-[200px]">
        <div className="flex justify-between items-center mb-2 border-b border-[var(--divider)] pb-1">
          <span className="text-[11px] font-bold text-[var(--text-primary)] font-mono">{data.time}</span>
          <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-widest">{isProjected ? 'Forecast' : 'Actual'}</span>
        </div>
        
        <div className="flex flex-col gap-1.5">
          {data.actual && (
            <div className="flex justify-between items-baseline">
              <span className="text-[10px] text-[var(--brand-teal)] font-bold uppercase">Actual</span>
              <span className="text-[14px] font-black">{Math.round(data.actual)} <small className="text-[10px] font-normal">MW</small></span>
            </div>
          )}
          {data.projected && (
            <div className="flex justify-between items-baseline">
              <span className="text-[10px] text-[var(--brand-blue-vibrant)] font-bold uppercase">Our Forecast</span>
              <span className="text-[14px] font-black">{Math.round(data.projected)} <small className="text-[10px] font-normal">MW</small></span>
            </div>
          )}
          {data.simday && (
            <div className="flex justify-between items-baseline">
              <span className="text-[10px] text-[#f39c12] font-bold uppercase">GRIDCo SimDay</span>
              <span className="text-[14px] font-black">{Math.round(data.simday)} <small className="text-[10px] font-normal">MW</small></span>
            </div>
          )}
        </div>

        {/* Embedded Explainability (Task C) */}
        {isProjected && (
          <div className="mt-3 pt-2 border-t border-[var(--divider)]">
            <div className="flex items-center gap-1.5 mb-2">
              <Info className="w-3 h-3 text-[var(--brand-gold)]" />
              <span className="text-[9px] font-bold uppercase tracking-tighter text-[var(--brand-gold)]">Dominant Drivers</span>
            </div>
            <div className="flex flex-col gap-1">
              {[
                { name: 'Temp Gradient', impact: '+4.2%' },
                { name: 'Lag (24h)', impact: '+2.1%' },
                { name: 'Humidity', impact: '-1.2%' }
              ].map((driver, idx) => (
                <div key={idx} className="flex justify-between text-[10px] font-mono">
                  <span className="text-[var(--text-muted)]">{driver.name}</span>
                  <span className={driver.impact.startsWith('+') ? 'text-[var(--status-emerald)]' : 'text-[var(--status-crimson)]'}>
                    {driver.impact}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }
  return null;
};

export function LoadChart({
  data,
  showUncertainty = true,
  liveMarkerTime,
  height = 360,
  viewType = 'default'
}: LoadChartProps) {
  if (viewType === 'stacked') {
    return (
      <div className="w-full" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="0" stroke="var(--chart-grid)" vertical={false} strokeWidth={0.5} />
            <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={11} tickMargin={10} fontFamily="JetBrains Mono" axisLine={false} tickLine={false} />
            <YAxis stroke="var(--text-muted)" fontSize={11} tickMargin={10} fontFamily="JetBrains Mono" axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--brand-teal)', strokeWidth: 1 }} />
            
            {/* GRIDCo Demand Stack */}
            <Area type="monotone" dataKey="domestic" stackId="1" stroke="#3498db" fill="#3498db" fillOpacity={0.6} name="Domestic (ECG)" />
            <Area type="monotone" dataKey="industrial" stackId="1" stroke="#e67e22" fill="#e67e22" fillOpacity={0.6} name="Industrial (Mines)" />
            <Area type="monotone" dataKey="exports" stackId="1" stroke="#9b59b6" fill="#9b59b6" fillOpacity={0.6} name="Exports (CEB/SONABEL)" />
            
            {liveMarkerTime && <ReferenceLine x={liveMarkerTime} stroke="var(--brand-gold)" strokeDasharray="4 4" />}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid 
            strokeDasharray="0" 
            stroke="var(--chart-grid)" 
            vertical={false} 
            strokeWidth={0.5}
          />
          
          <XAxis 
            dataKey="time" 
            stroke="var(--text-muted)" 
            fontSize={11} 
            tickMargin={10} 
            fontFamily="JetBrains Mono"
            axisLine={false}
            tickLine={false}
          />
          
          <YAxis 
            stroke="var(--text-muted)" 
            fontSize={11} 
            tickMargin={10} 
            fontFamily="JetBrains Mono"
            axisLine={false}
            tickLine={false}
            label={{ 
              value: 'MW', 
              angle: -90, 
              position: 'insideLeft', 
              style: { fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--text-muted)' } 
            }}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--brand-teal)', strokeWidth: 1, strokeDasharray: '4 4' }} />

          {/* Uncertainty Band - use range array for proper area */}
          {showUncertainty && (
            <Area
              type="monotone"
              dataKey="range"
              data={data.map(d => ({ ...d, range: d.lowerBound && d.upperBound ? [d.lowerBound, d.upperBound] : null }))}
              fill="var(--brand-teal)"
              fillOpacity={0.15}
              stroke="var(--brand-teal)"
              strokeWidth={0}
              connectNulls
            />
          )}

          {/* Live Marker */}
          {liveMarkerTime && (
            <ReferenceLine 
              x={liveMarkerTime} 
              stroke="var(--brand-gold)" 
              strokeDasharray="4 4" 
              label={{ 
                position: 'top', 
                value: '● LIVE', 
                fill: 'var(--brand-gold)', 
                fontSize: 10, 
                fontWeight: 'bold',
                fontFamily: 'JetBrains Mono' 
              }} 
            />
          )}

          {/* Baseline Curve */}
          <Line 
            type="monotone" 
            dataKey="baseline" 
            stroke="var(--text-muted)" 
            strokeWidth={1.5} 
            strokeDasharray="6 4" 
            dot={false} 
            activeDot={false}
            connectNulls
          />

          {/* Actual Load Curve (History context) */}
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="var(--brand-teal)" 
            strokeWidth={2.5} 
            dot={false} 
            activeDot={{ r: 4, fill: 'var(--brand-teal)', strokeWidth: 2, stroke: 'var(--bg-card)' }}
            connectNulls
          />

          {/* GRIDCo Similar Day Curve */}
          <Line 
            type="monotone" 
            dataKey="simday" 
            stroke="#f39c12" 
            strokeWidth={2} 
            strokeDasharray="4 4"
            dot={false} 
            activeDot={{ r: 3, fill: '#f39c12', strokeWidth: 1, stroke: 'var(--bg-card)' }}
            connectNulls
          />

          {/* Projected Load Curve */}
          <Line 
            type="monotone" 
            dataKey="projected" 
            stroke="var(--brand-blue-vibrant)" 
            strokeWidth={3} 
            dot={false} 
            activeDot={{ r: 4, fill: 'var(--brand-blue-vibrant)', strokeWidth: 2, stroke: 'var(--bg-card)' }}
            connectNulls
            animationDuration={800}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
