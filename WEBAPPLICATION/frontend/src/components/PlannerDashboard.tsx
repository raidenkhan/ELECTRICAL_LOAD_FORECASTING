'use client';

import React, { useState, useEffect } from 'react';
import { WeeklyForecast } from './WeeklyForecast';
import { MonthlyComparison } from './MonthlyComparison';
import { ScenarioSimulator } from './ScenarioSimulator';
import { forecastService, ForecastResponse } from '@/services/forecastService';

export function PlannerDashboard() {
  const [ltlfData, setLtlfData] = useState<ForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [horizonDays, setHorizonDays] = useState(30);

  useEffect(() => {
    async function fetchLTLF() {
      try {
        setIsLoading(true);
        // Fetch requested horizon (days * 24 h)
        const data = await forecastService.getLTLF(horizonDays * 24);
        setLtlfData(data);
      } catch (error) {
        console.error('Failed to fetch LTLF data:', error);
      } finally {
        setIsLoading(false);
      }
    }
    fetchLTLF();
  }, [horizonDays]);

  return (
    <div className="flex flex-col gap-10 h-full font-sans">

      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="headline text-[var(--text-primary)]">System Planning & Analysis</h1>
          <p className="body-text text-[var(--text-secondary)]">
            Manage forward-looking load projections, seasonal comparisons, and scenario-based capacity planning.
          </p>
        </div>
        
        {/* Horizon Selector */}
        <div className="flex items-center gap-1 bg-[var(--surface-secondary)]/60 p-1 rounded-sm border border-[var(--divider)]">
          {[
            { label: '1 Week', value: 7 },
            { label: '1 Month', value: 30 },
            { label: '1 Quarter', value: 90 },
            { label: '1 Year', value: 365 }
          ].map(opt => (
             <button
                key={opt.value}
                onClick={() => setHorizonDays(opt.value)}
                className={`px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider rounded-sm transition-colors ${
                  horizonDays === opt.value 
                  ? 'bg-[var(--brand-blue)] text-white' 
                  : 'text-[var(--text-primary)] hover:bg-[var(--divider)]/50'
                }`}
             >
                {opt.label}
             </button>
          ))}
        </div>
      </div>

      {/* Horizon Outlook */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">Horizon Outlook</h2>
          <span className="caption text-[var(--text-muted)]">Selected Horizon | Recursive Model</span>
        </div>
        <WeeklyForecast data={ltlfData} isLoading={isLoading} />
      </section>

      {/* Comparison & Simulation Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">

        {/* Monthly Trend Comparison */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">Trend Scaling</h2>
            <span className="caption text-[var(--text-muted)]">YoY Comparison</span>
          </div>
          <MonthlyComparison data={ltlfData} isLoading={isLoading} />
        </section>

        {/* What-If Scenario Builder */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">GRIDCo What-If Builder</h2>
            <span className="caption text-[var(--text-muted)]">Policy & Weather Simulations</span>
          </div>
          <ScenarioSimulator 
            defaultPeak={ltlfData?.forecast_mw ? Math.max(...ltlfData.forecast_mw) : 155} 
          />
        </section>

      </div>
    </div>
  );
}
