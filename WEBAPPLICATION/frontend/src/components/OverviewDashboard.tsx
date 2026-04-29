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
import { forecastService, ForecastResponse } from '@/services/forecastService';
import { useSystem } from '@/context/SystemContext';


export function OverviewDashboard() {
  const { updateSync } = useSystem();
  const [showExplainability, setShowExplainability] = useState(false);
  const [stlf, setStlf] = useState<ForecastResponse | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [horizonHours, setHorizonHours] = useState(24);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        setError(null);
        // Always fetch max horizon (24H) to allow instant local switching
        const [stlfData, metricsData] = await Promise.all([
          forecastService.getSTLF(24), 
          forecastService.getModelMetrics()
        ]);
        setStlf(stlfData);
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
  }, [updateSync]); // Removed horizonHours from dependencies

  // Local filtering logic for Task A optimization
  const chartData = stlf && stlf.forecast_mw?.length > 0 ? stlf.timestamps.map((t, i) => ({
    time: new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    actual: i === 0 ? stlf.forecast_mw[i] : null,
    projected: stlf.forecast_mw[i],
    simday: stlf.simday_forecast_mw ? stlf.simday_forecast_mw[i] : null,
    baseline: stlf.forecast_mw[i] * 0.98,
    upperBound: stlf.p90 ? stlf.p90[i] : (stlf.forecast_mw[i] * 1.1),
    lowerBound: stlf.p10 ? stlf.p10[i] : (stlf.forecast_mw[i] * 0.9),
  })).slice(0, (horizonHours * 4) + 1) : []; // +1 to show current moment. (*4 because 15m intervals)

  const currentLoad = stlf && stlf.forecast_mw?.length > 0 ? Math.round(stlf.forecast_mw[0]) : 124;
  const peakLoad = stlf && stlf.forecast_mw?.length > 0 ? Math.round(Math.max(...stlf.forecast_mw)) : 158;
  const peakTime = stlf && stlf.forecast_mw?.length > 0
    ? new Date(stlf.timestamps[stlf.forecast_mw.indexOf(Math.max(...stlf.forecast_mw))]).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '19:30';

  const regimeData = stlf?.regime_distribution ?? [];
  const accuracy = metrics?.summary?.find((m: any) => m.horizon.includes('STLF'))?.r_squared 
    ? (metrics.summary.find((m: any) => m.horizon.includes('STLF')).r_squared * 100).toFixed(1)
    : "96.8";

  // Build a readable forecast window label, e.g. "08:30 → +24H"
  const forecastStart = stlf && stlf.timestamps.length > 0
    ? new Date(stlf.timestamps[0]).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const forecastWindowLabel = `${forecastStart} → +${horizonHours}H`;

  // Smart peak label: handles today, tomorrow (24H window crosses midnight), or stale
  const peakDate = stlf && stlf.timestamps.length > 0
    ? new Date(stlf.timestamps[stlf.forecast_mw.indexOf(Math.max(...stlf.forecast_mw))])
    : null;
  const peakSubtext = (() => {
    if (!peakDate) return 'Calculating...';
    const now = new Date();
    const todayStr = now.toDateString();
    const tomorrowStr = new Date(now.getTime() + 86400000).toDateString();
    if (peakDate < now) return 'Peak within forecast window';
    if (peakDate.toDateString() === todayStr) return `Expected at ${peakTime}`;
    if (peakDate.toDateString() === tomorrowStr) return `Expected tomorrow at ${peakTime}`;
    return `Expected ${peakDate.toLocaleDateString([], { weekday: 'short' })} at ${peakTime}`;
  })();

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
          Regional Grid / Nayagina-82 / Real-time sync active
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard label="Estimated Load" value={isLoading ? "..." : currentLoad.toLocaleString()} unit="MW" status="emerald" subtext={`T+15m · window: ${forecastWindowLabel}`} />
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
            <h2 className="headline text-[var(--text-primary)]">Short-Term Load Forecast (STLF)</h2>
            <p className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5 uppercase tracking-wider">
              {isLoading ? 'Calculating window...' : `Forecast window: ${forecastWindowLabel}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex bg-[var(--bg-secondary)] rounded-lg p-1 border border-[var(--border-primary)]">
                {[6, 12, 24].map((h) => (
                    <button
                        key={h}
                        onClick={() => setHorizonHours(h)}
                        className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
                            horizonHours === h 
                            ? 'bg-[var(--brand-blue-vibrant)] text-white shadow-md' 
                            : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                        }`}
                    >
                        {h}H
                    </button>
                ))}
            </div>
            <StatusBadge status={isLoading ? "connecting" : "stable"} label={isLoading ? "Syncing..." : "Live"} pulse={isLoading} />
          </div>
        </div>
        <LoadChart 
          data={chartData} 
          liveMarkerTime={
            stlf && stlf.timestamps.length > 0 
            ? new Date(stlf.timestamps[0]).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
            : '--:--'
          } 
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        <div className="xl:col-span-2 glass-panel p-6">
           <div className="flex justify-between items-center mb-6">
              <h2 className="title text-[var(--text-primary)] uppercase tracking-widest">Operating Regime Probabilities</h2>
               <button onClick={() => setShowExplainability(true)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
                <Info className="w-4 h-4" />
              </button>
           </div>
           
           <div className="h-[300px] w-full">
            {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <span className="micro-num animate-pulse tracking-widest">DECOMPOSING SIGNAL...</span>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={regimeData} barGap={8}>
                    <CartesianGrid strokeDasharray="0" stroke="var(--divider)" vertical={false} />
                    <XAxis dataKey="hour" stroke="var(--text-muted)" fontSize={11} fontFamily="JetBrains Mono" />
                    <YAxis stroke="var(--text-muted)" fontSize={11} fontFamily="JetBrains Mono" tickFormatter={(v) => `${v}%`} />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-card)', borderRadius: '4px' }} cursor={{ fill: 'var(--divider)', opacity: 0.3 }} />
                    <Bar dataKey="regime0" stackId="a" fill="var(--status-emerald)" name="Standard" />
                    <Bar dataKey="regime1" stackId="a" fill="var(--status-amber)" name="Transition" />
                    <Bar dataKey="regime2" stackId="a" fill="var(--status-crimson)" name="Peak" />
                  </BarChart>
                </ResponsiveContainer>
              )}
           </div>
        </div>

        <div className="glass-panel p-6">
          <h2 className="title text-[var(--text-primary)] uppercase tracking-widest mb-6">Dispatch Alerts</h2>
          <ActiveAlerts />
        </div>
      </div>

      <ExplainabilityModal
        isOpen={showExplainability}
        onClose={() => setShowExplainability(false)}
      />
    </div>
  );
}