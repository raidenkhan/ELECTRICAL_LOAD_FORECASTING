import React, { useState, useMemo } from 'react';
import { MetricCard } from './MetricCard';
import { StatusBadge } from './StatusBadge';
import { LoadChart } from './LoadChart';
import { GenerationUnit } from './GenerationUnit';
import { AlertBanner } from './AlertBanner';
import { Layers, TrendingUp, Zap, Wind, HardHat, Globe } from 'lucide-react';

// Enhanced Mock data generator for the Power Balance Planner
const generatePlannerData = (temp: number, use8PercentRule: boolean) => {
  return Array.from({ length: 24 }).map((_, i) => {
    const time = `${i.toString().padStart(2, '0')}:00`;
    
    // Core physics: Domestic load is 90% and weather-driven
    const baseDomestic = 85 + Math.sin(i / 12 * Math.PI) * 25;
    const tempEffect = Math.max(0, temp - 24) * 2.6; // 2.6 coefficient from interview
    let domestic = baseDomestic + tempEffect;
    
    // Apply GRIDCo Strategic Rule (8% YoY increment) if active
    if (use8PercentRule) {
      domestic = domestic * 1.08;
    }

    // Industrial load (Flat-ish as mentioned in interview: VALCO 100MW + Mines)
    const industrial = 120 + (Math.random() * 5); 
    
    // Exports (CEB/SONABEL mentioned as ~347MW total)
    const exports = 347 + (Math.random() * 10 - 5);

    const projected = domestic + industrial + exports;
    
    return {
      time,
      domestic,
      industrial,
      exports,
      projected,
      baseline: projected * 0.95
    };
  });
};

export function DigitalTwinDashboard() {
  const [temperature, setTemperature] = useState(32);
  const [plannerMode, setPlannerMode] = useState<'balance' | 'strategic'>('balance');
  const [use8PercentRule, setUse8PercentRule] = useState(false);
  
  const chartData = useMemo(() => generatePlannerData(temperature, use8PercentRule), [temperature, use8PercentRule]);
  
  const totalDemand = chartData[14]?.projected || 0;
  const domesticLoad = chartData[14]?.domestic || 0;
  
  // Real-time Elasticity Calculation (800MW drop mentioned for 5C change)
  const elasticityPerDegree = 160; // 800 / 5 = 160 MW/C
  const potentialSavings = (temperature - 27) * elasticityPerDegree;

  return (
    <div className="flex flex-col gap-6 h-full font-sans">
      
      {/* 3-Column Workspace */}
      <div className="flex flex-1 min-h-[500px]">
        
        {/* LEFT PANEL: Power Balance Controls */}
        <div className="w-[300px] flex-shrink-0 flex flex-col glass-panel p-5 mr-6">
          <div className="flex flex-col gap-8">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-black text-[var(--brand-blue)] uppercase tracking-[0.3em]">Operational Planning</span>
              <h3 className="text-[18px] font-bold text-[var(--text-primary)]">Power Balance</h3>
            </div>
            
            {/* Instant Temperature Pivot */}
            <div className="flex flex-col py-2">
              <span className="text-[11px] font-black text-[var(--text-muted)] uppercase tracking-widest mb-4">Thermal Pivot (27°C - 32°C)</span>
              <div className="p-4 bg-[var(--surface-secondary)]/60 rounded border border-[var(--divider)] mb-4 text-center">
                  <span className="text-3xl font-black text-[var(--text-primary)]">{temperature}°C</span>
                  <div className="flex justify-between mt-2 micro-num opacity-40 uppercase">
                    <span>Wet/Rain</span>
                    <span>High Peak</span>
                  </div>
              </div>
              
              <input 
                type="range" min="24" max="40" step="0.5"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full transition-all accent-[var(--brand-blue)]"
              />
              
              <div className="mt-6 p-3 bg-blue-500/10 border border-blue-500/20 rounded flex items-center gap-3">
                 <Wind className="w-5 h-5 text-blue-400" />
                 <div>
                    <p className="text-[10px] font-bold uppercase text-blue-300">Cooling Gain</p>
                    <p className="text-lg font-black text-white">-{Math.round(potentialSavings)} MW</p>
                 </div>
              </div>
            </div>

            {/* Strategic Rule Toggle */}
            <div className="flex flex-col gap-3">
              <span className="text-[11px] font-black text-[var(--text-muted)] uppercase tracking-widest">Strategic Horizon</span>
              <button 
                onClick={() => setUse8PercentRule(!use8PercentRule)}
                className={`w-full p-4 rounded border flex flex-col items-center gap-1 transition-all
                  ${use8PercentRule ? 'bg-indigo-600 border-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.3)]' : 'bg-[var(--surface-secondary)] border-[var(--divider)] hover:border-[var(--brand-blue)]'}
                `}
              >
                <TrendingUp className={`w-5 h-5 ${use8PercentRule ? 'text-white' : 'text-[var(--brand-blue)]'}`} />
                <span className={`text-[11px] font-black uppercase ${use8PercentRule ? 'text-white' : 'text-[var(--text-primary)]'}`}>
                   GRIDCo 8% Growth Rule
                </span>
                <span className="text-[9px] text-white/50 uppercase tracking-tighter">Applied to Domestic Base</span>
              </button>
            </div>

            {/* Demand Category Distribution */}
            <div className="flex flex-col gap-3 mt-4">
              <span className="text-[11px] font-black text-[var(--text-muted)] uppercase tracking-widest">Demand Composition</span>
              <div className="space-y-2">
                 <CategoryBadge label="Domestic (ECG)" value="90%" icon={<Globe className="w-3 h-3"/>} color="#3498db" />
                 <CategoryBadge label="Industrial (Mines)" value="~120MW" icon={<HardHat className="w-3 h-3"/>} color="#e67e22" />
                 <CategoryBadge label="Exports" value="347MW" icon={<Zap className="w-3 h-3"/>} color="#9b59b6" />
              </div>
            </div>
          </div>
        </div>

        {/* CENTER PANEL: Stacked Load Planner */}
        <div className="flex-1 min-w-0 flex flex-col glass-panel p-5 overflow-hidden mx-0 border-t-4 border-t-[var(--brand-blue)]">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
               <Layers className="w-4 h-4 text-[var(--brand-blue)]" />
               <h2 className="headline text-[var(--text-primary)] uppercase tracking-widest text-[12px] font-black">System Balance Planner</h2>
            </div>
            <div className="flex gap-2">
               <button 
                 onClick={() => setPlannerMode('balance')}
                 className={`px-4 py-1.5 rounded-sm text-[10px] font-black uppercase tracking-widest border transition-all
                   ${plannerMode === 'balance' ? 'bg-[var(--brand-blue)] border-[var(--brand-blue)] text-white' : 'border-[var(--divider)] text-[var(--text-muted)]'}
                 `}
               >Demand Stack</button>
               <button 
                 onClick={() => setPlannerMode('strategic')}
                 className={`px-4 py-1.5 rounded-sm text-[10px] font-black uppercase tracking-widest border transition-all
                   ${plannerMode === 'strategic' ? 'bg-[var(--brand-blue)] border-[var(--brand-blue)] text-white' : 'border-[var(--divider)] text-[var(--text-muted)]'}
                 `}
               >Forecast View</button>
            </div>
          </div>
          
          <div className="flex-1 min-h-0">
            <LoadChart 
              data={chartData} 
              liveMarkerTime="14:00" 
              viewType={plannerMode === 'balance' ? 'stacked' : 'default'}
              height={440}
            />
          </div>

          <div className="mt-6 p-4 bg-[var(--surface-secondary)]/40 border border-[var(--divider)] rounded flex items-center justify-between">
              <div className="flex items-center gap-6">
                 <div className="flex flex-col">
                    <span className="caption text-[var(--text-muted)] uppercase font-black">System Total</span>
                    <span className="text-xl font-black text-[var(--text-primary)]">{Math.round(totalDemand)} MW</span>
                 </div>
                 <div className="w-px h-8 bg-[var(--divider)]" />
                 <div className="flex flex-col">
                    <span className="caption text-[var(--text-muted)] uppercase font-black text-[#3498db]">Domestic</span>
                    <span className="text-xl font-black text-[#3498db]">{Math.round(domesticLoad)} MW</span>
                 </div>
              </div>
              <div className="text-right">
                 <p className="text-[10px] text-[var(--text-muted)] italic leading-tight max-w-[200px]">
                    "90% dependent on what happens in our homes" — NCC Planning Interview
                 </p>
              </div>
          </div>
        </div>

        {/* RIGHT PANEL: Balance Intelligence */}
        <div className="w-[320px] flex-shrink-0 flex flex-col glass-panel p-5 ml-6">
           <h2 className="headline text-[var(--text-secondary)] uppercase tracking-widest text-[11px]">Planning Intelligence</h2>
           
           <div className="space-y-6">
              <StatusBadge status={temperature > 35 ? 'critical' : 'stable'} label={temperature > 35 ? 'Heat Stress Threshold' : 'Stable Operating Regime'} pulse />

              <div className="h-px bg-[var(--divider)]" />

              {/* GRIDCo Physical Knot Analysis */}
              <div className="space-y-4">
                 <p className="text-[11px] font-bold text-[var(--text-primary)] uppercase flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-yellow-500" /> Physical Knot Analysis
                 </p>
                 <div className="space-y-3">
                    <KnotItem label="AC Saturation Point" value="24.0°C" status="active" />
                    <KnotItem label="Thermal Elasticity" value="2.6 MW/°C" status="active" />
                    <KnotItem label="Line Efficiency Gain" value={temperature < 22 ? "+1.5%" : "Baseline"} status={temperature < 22 ? 'active' : 'idle'} />
                 </div>
              </div>

              <div className="h-px bg-[var(--divider)]" />

              {/* Reserve Margin Monitor */}
              <div className="p-4 bg-[var(--surface-secondary)] border border-[var(--divider)] rounded-lg">
                <span className="caption text-[var(--text-muted)] uppercase tracking-widest font-bold">Projected Reserve</span>
                <div className="flex items-baseline gap-2 mt-1">
                  <span className={`metric-num text-5xl font-bold leading-none
                    ${totalDemand > 1800 ? 'text-[var(--status-crimson)] animate-pulse' : 'text-emerald-500'}
                  `}>
                    {Math.round(2100 - totalDemand)}
                  </span>
                  <span className="text-[16px] font-bold text-[var(--text-secondary)] uppercase">MW</span>
                </div>
                <div className="h-1.5 w-full bg-gray-800 rounded-full mt-4 overflow-hidden">
                   <div className="h-full bg-emerald-500" style={{ width: `${Math.max(0, (2100 - totalDemand) / 2100 * 100)}%` }} />
                </div>
              </div>

              <AlertBanner 
                severity={totalDemand > 1800 ? 'critical' : 'info'}
                message={totalDemand > 1800 ? 'Insufficient spinning reserve at peak' : 'Generation fleet meets demand stack requirements'}
              />
           </div>
        </div>
      </div>
    </div>
  );
}

function CategoryBadge({ label, value, icon, color }: { label: string, value: string, icon: React.ReactNode, color: string }) {
  return (
    <div className="flex items-center justify-between p-2 rounded border border-white/5 bg-white/5">
       <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded flex items-center justify-center" style={{ backgroundColor: `${color}20`, color: color }}>
             {icon}
          </div>
          <span className="text-[10px] font-bold text-[var(--text-secondary)] uppercase tracking-tight">{label}</span>
       </div>
       <span className="text-[10px] font-black font-mono text-[var(--text-primary)]">{value}</span>
    </div>
  );
}

function KnotItem({ label, value, status }: { label: string, value: string, status: 'active' | 'idle' }) {
  return (
    <div className="flex justify-between items-center">
       <span className="text-[11px] text-[var(--text-muted)]">{label}</span>
       <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono font-bold text-[var(--text-primary)]">{value}</span>
          <div className={`w-1.5 h-1.5 rounded-full ${status === 'active' ? 'bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.5)]' : 'bg-gray-600'}`} />
       </div>
    </div>
  );
}
