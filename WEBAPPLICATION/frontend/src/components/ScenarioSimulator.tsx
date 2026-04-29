import React, { useState } from 'react';

export function ScenarioSimulator({ defaultPeak = 155 }: { defaultPeak?: number }) {
  const [tempDelta, setTempDelta] = useState(0);
  const [yoyGrowth, setYoyGrowth] = useState(8.5);
  const [industrialLoad, setIndustrialLoad] = useState(0);

  const basePeak = defaultPeak;
  // Simplified sensitivity logic for simulation
  const adjustedPeak = basePeak * (1 + (yoyGrowth - 8.5) / 100) + (tempDelta * (basePeak / 10)) + (industrialLoad * (basePeak / 50));
  const delta = adjustedPeak - basePeak;

  return (
    <div className="glass-panel p-6 flex flex-col gap-8 h-full">
      <div className="space-y-6">
        {/* Slider: Temperature */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-[13px] font-bold text-[var(--text-primary)]">Temperature Variance</span>
            <span className={`data-num font-bold ${tempDelta > 0 ? 'text-[var(--status-amber)]' : 'text-[var(--brand-teal)]'}`}>
              {tempDelta > 0 ? '+' : ''}{tempDelta}°C
            </span>
          </div>
          <input
            type="range" min="-5" max="5" step="0.5"
            value={tempDelta}
            onChange={(e) => setTempDelta(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-[var(--divider)] rounded-full appearance-none cursor-pointer accent-[var(--brand-indigo)]"
          />
          <div className="flex justify-between text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
            <span>Cooler</span>
            <span>Relative to Forecast</span>
            <span>Hotter</span>
          </div>
        </div>

        {/* Slider: YoY Growth */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-[13px] font-bold text-[var(--text-primary)]">YoY Growth Rate</span>
            <span className="data-num font-bold text-[var(--brand-blue)]">{yoyGrowth}%</span>
          </div>
          <input
            type="range" min="6" max="14" step="0.5"
            value={yoyGrowth}
            onChange={(e) => setYoyGrowth(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-[var(--divider)] rounded-full appearance-none cursor-pointer accent-[var(--brand-indigo)]"
          />
          <div className="flex justify-between text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
            <span>6%</span>
            <span>Target: 8.5%</span>
            <span>14%</span>
          </div>
        </div>

        {/* Slider: Industrial Load */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-[13px] font-bold text-[var(--text-primary)]">Industrial Demand</span>
            <span className="data-num font-bold text-[var(--brand-blue)]">
              {industrialLoad > 0 ? '+' : ''}{industrialLoad}%
            </span>
          </div>
          <input
            type="range" min="-20" max="20" step="1"
            value={industrialLoad}
            onChange={(e) => setIndustrialLoad(parseInt(e.target.value))}
            className="w-full h-1.5 bg-[var(--divider)] rounded-full appearance-none cursor-pointer accent-[var(--brand-indigo)]"
          />
        </div>
      </div>

      {/* Results Panel */}
      <div className="mt-auto p-5 bg-[var(--brand-indigo)]/5 border border-[var(--brand-indigo)]/20 rounded-lg space-y-4">
        <div className="flex justify-between items-center border-b border-[var(--brand-indigo)]/10 pb-3">
          <span className="caption text-[var(--text-muted)] uppercase tracking-widest font-bold">Base Forecast Peak</span>
          <span className="data-num text-[var(--text-primary)] font-bold">{basePeak} MW</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="caption text-[var(--brand-blue)] uppercase tracking-widest font-bold">Adjusted Projection</span>
          <div className="flex flex-col items-end">
            <span className="metric-num text-3xl text-[var(--brand-blue)]">{Math.round(adjustedPeak)}</span>
            <span className={`text-[11px] font-bold ${delta >= 0 ? 'text-[var(--status-amber)]' : 'text-[var(--status-emerald)]'}`}>
              {delta >= 0 ? '+' : ''}{Math.round(delta)} MW Delta
            </span>
          </div>
        </div>

        <div className="h-px bg-[var(--brand-indigo)]/10" />

        <div className="flex gap-3 items-center">
          <div className="w-2 h-2 rounded-full bg-[var(--brand-indigo)] animate-pulse" />
          <p className="text-[12px] font-medium text-[var(--text-primary)]">
            {adjustedPeak > 1650
              ? 'Recommended: Ensure KPONE is on standby for peak window.'
              : 'Projected margin stable within reserve limits.'}
          </p>
        </div>
      </div>
    </div>
  );
}

