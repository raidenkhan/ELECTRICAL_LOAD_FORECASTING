'use client';

import React, { useState, useEffect } from 'react';
import { WeeklyForecast } from './WeeklyForecast';
import { MonthlyComparison } from './MonthlyComparison';
import { ScenarioSimulator } from './ScenarioSimulator';
import { dispatchForecastService, DispatchForecastResponse } from '@/services/dispatchForecastService';

export function PlannerDashboard() {
  const [forecastData, setForecastData] = useState<DispatchForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchDispatchForecast() {
      try {
        setIsLoading(true);
        const data = await dispatchForecastService.getTomorrow();
        setForecastData(data);
      } catch (error) {
        console.error('Failed to fetch dispatch forecast:', error);
      } finally {
        setIsLoading(false);
      }
    }
    fetchDispatchForecast();
  }, []);

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
          <span className="px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-[var(--brand-blue)]">
            D+1 (24H)
          </span>
        </div>
      </div>

      {/* Horizon Outlook */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">Horizon Outlook</h2>
          <span className="caption text-[var(--text-muted)]">DLinear+TIDE | 6-Fold Ensemble</span>
        </div>
        <WeeklyForecast data={forecastData} isLoading={isLoading} />
      </section>

      {/* Comparison & Simulation Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">

        {/* Monthly Trend Comparison */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">Trend Scaling</h2>
            <span className="caption text-[var(--text-muted)]">YoY Comparison</span>
          </div>
          <MonthlyComparison data={forecastData} isLoading={isLoading} />
        </section>

        {/* What-If Scenario Builder */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold">GRIDCo What-If Builder</h2>
            <span className="caption text-[var(--text-muted)]">Policy & Weather Simulations</span>
          </div>
          <ScenarioSimulator 
            defaultPeak={forecastData?.forecast_mw ? Math.max(...forecastData.forecast_mw) : 155} 
          />
        </section>

      </div>
    </div>
  );
}
