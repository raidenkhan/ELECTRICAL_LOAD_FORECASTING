'use client';

import {
  TrendingUp,
  Activity,
  Zap,
  AlertTriangle,
  Info
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { MetricCard } from './MetricCard';
import { StatusBadge } from './StatusBadge';
import { LoadChart } from './LoadChart';
import { ActiveAlerts } from './ActiveAlerts';
import { useState, useEffect } from 'react';
import { ExplainabilityModal } from './ExplainabilityModal';
import { dispatchForecastService, DispatchForecastResponse } from '@/services/dispatchForecastService';
import { forecastService } from '@/services/forecastService';
import { useSystem } from '@/context/SystemContext';


export function OverviewDashboard() {
  const { updateSync } = useSystem();
  const [showExplainability, setShowExplainability] = useState(false);
  const [forecast, setForecast] = useState<DispatchForecastResponse | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        setError(null);
        const [forecastData, metricsData] = await Promise.all([
          dispatchForecastService.getTomorrow(),
          forecastService.getModelMetrics()
        ]);
        setForecast(forecastData);
        setMetrics(metricsData);
        updateSync();
      } catch (err: any) {
        console.error('Failed to fetch dashboard data:', err);
        setError(err.message || 'Failed to load system data');
      } finally {
        setIsLoading(false);
      }
    }
    fetchData();
  }, [updateSync]);

  const chartData = forecast && forecast.forecast_mw?.length > 0
    ? forecast.forecast_mw.map((mw, i) => ({
        time: `${i + 1}h`,
        actual: i === 0 ? mw : null,
        projected: mw,
        baseline: mw * 0.98,
        upperBound: forecast.p90_mw ? forecast.p90_mw[i] : (mw * 1.1),
        lowerBound: forecast.p10_mw ? forecast.p10_mw[i] : (mw * 0.9),
      }))
    : [];

  const currentLoad = forecast && forecast.forecast_mw?.length > 0 ? Math.round(forecast.forecast_mw[0]) : 124;
  const peakLoad = forecast && forecast.forecast_mw?.length > 0 ? Math.round(Math.max(...forecast.forecast_mw)) : 158;
  const peakHour = forecast && forecast.forecast_mw?.length > 0
    ? forecast.forecast_mw.indexOf(Math.max(...forecast.forecast_mw)) + 1
    : 19;
  const peakTime = `H${peakHour.toString().padStart(2, '0')}:00`;

  const accuracy = metrics?.db_metrics?.find((m: any) => m.horizon === '24h')?.mape
    ? (100 - metrics.db_metrics.find((m: any) => m.horizon === '24h').mape).toFixed(1)
    : "96.3";

  const forecastWindowLabel = 'D+1 (24H)';

  const peakDate = forecast && forecast.forecast_mw?.length > 0
    ? forecast.forecast_date
    : null;
  const peakSubtext = peakDate
    ? `Expected ${peakDate} at ${peakTime}`
    : 'Calculating...';

  const isHealthy = peakLoad < 140;
  const isWarning = peakLoad >= 140 && peakLoad < 150;
  const isCritical = peakLoad >= 150;
  
  const badgeStatus = isCritical ? 'critical' : isWarning ? 'warning' : 'stable';
  const badgeLabel = isCritical ? 'Risk Level High' : isWarning ? 'Approaching Limit' : 'Operational';
  const marginValue = 150 - peakLoad;
  const marginText = isCritical 
    ? `Deficit (${Math.abs(marginValue)} MW)` 
    : `Margin (${marginValue > 0 ? marginValue : 0} MW)`;

  return (
    <div className="flex flex-col gap-10">
      {error && (
        <div className="p-4 bg-[var(--status-crimson)]/10 border border-[var(--status-crimson)]/30 text-[var(--status-crimson)] rounded flex items-center gap-3">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-bold uppercase text-[12px] tracking-tight">{error}</span>
        </div>
      )}

      {/* Primary Header */}
      <div className="flex flex-col gap-1">
        <h1 className="display-num text-[var(--text-primary)]">System Overview</h1>
        <p className="caption text-[var(--text-muted)] uppercase tracking-[0.3em]">
          GRIDCo National Grid / Real-time sync active
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard label="Estimated Load" value={isLoading ? "..." : currentLoad.toLocaleString()} unit="MW" status="emerald" subtext={`window: ${forecastWindowLabel}`} />
        <MetricCard label="Accuracy Index" value={isLoading ? "..." : accuracy} unit="%" status="emerald" subtext="R² Stability Index" trend={{ value: 0.3, isUp: true }} />
        <MetricCard label="Today's Peak" value={isLoading ? "..." : peakLoad.toLocaleString()} unit="MW" subtext={isLoading ? '...' : peakSubtext} />
         <div className="glass-panel p-5 flex flex-col justify-between">
          <span className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Load Regime</span>
          <div className="mt-2 text-xs font-bold uppercase">
             {isLoading ? <span className="animate-pulse">Analyzing...</span> : <StatusBadge status={badgeStatus} label={badgeLabel} pulse={isCritical} />}
          </div>
          <p className="text-[11px] text-[var(--text-muted)] mt-2 italic font-medium">
            {isLoading ? "Calculating limit..." : marginText}
          </p>
        </div>
      </div>

      <div className="glass-panel p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="headline text-[var(--text-primary)]">Dispatch Forecast (DLinear+TIDE)</h2>
            <p className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5 uppercase tracking-wider">
              {isLoading ? 'Calculating window...' : `Forecast: ${forecastWindowLabel}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge status={isLoading ? "connecting" : "stable"} label={isLoading ? "Syncing..." : "Live"} pulse={isLoading} />
          </div>
        </div>
        <LoadChart 
          data={chartData} 
          liveMarkerTime={forecast?.forecast_date ?? '--:--'}
        />
      </div>

      <div className="glass-panel p-6">
        <h2 className="title text-[var(--text-primary)] uppercase tracking-widest mb-6">Dispatch Alerts</h2>
        <ActiveAlerts />
      </div>

      <ExplainabilityModal
        isOpen={showExplainability}
        onClose={() => setShowExplainability(false)}
      />
    </div>
  );
}