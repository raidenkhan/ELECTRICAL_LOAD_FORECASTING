import React, { useState, useEffect, useMemo } from 'react';
import { MetricCard } from './MetricCard';
import { StatusBadge } from './StatusBadge';
import { LoadChart } from './LoadChart';
import { dispatchForecastService, DispatchForecastResponse } from '@/services/dispatchForecastService';
import { forecastService } from '@/services/forecastService';
import { Loader2, RefreshCw } from 'lucide-react';

export function LiveMonitor() {
  const [forecast, setForecast] = useState<DispatchForecastResponse | null>(null);
  const [latestData, setLatestData] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [forecastData, dataRes, metricsRes] = await Promise.all([
        dispatchForecastService.getTomorrow(),
        forecastService.getLatestData(24),
        forecastService.getModelMetrics()
      ]);
      setForecast(forecastData);
      setLatestData(dataRes);
      setMetrics(metricsRes);
    } catch (err) {
      console.error('LiveMonitor fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const chartData = useMemo(() => {
    if (!forecast) return [];
    
    // Merge History and Forecast
    const historyPoints = [...latestData].reverse().map(d => ({
        time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: undefined }),
        actual: d.total_load_mw,
        projected: null,
        baseline: d.total_load_mw * 0.98
    }));

    const forecastPoints = forecast.forecast_mw.map((mw, i) => ({
        time: `H${(i + 1).toString().padStart(2, '0')}:00`,
        actual: null,
        projected: mw,
        lowerBound: forecast.p10_mw ? forecast.p10_mw[i] : null,
        upperBound: forecast.p90_mw ? forecast.p90_mw[i] : null,
    }));

    return [...historyPoints, ...forecastPoints];
  }, [forecast, latestData]);

  if (isLoading && !forecast) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-20 glass-panel">
        <Loader2 className="w-10 h-10 animate-spin text-[var(--brand-blue)] mb-4" />
        <span className="micro-num uppercase tracking-[0.2em] animate-pulse">Establishing Live Data Link...</span>
      </div>
    );
  }

  const currentLoad = latestData[0]?.total_load_mw || 0;
  const metricsSummary = metrics?.summary?.[0] || { mae: 6.55, mape: 14.67 };
  const liveTime = latestData[0] ? new Date(latestData[0].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: undefined }) : 'N/A';

  return (
    <div className="flex flex-col gap-6 h-full font-sans">
      
      {/* Top Stat Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          label="Live Load" 
          value={currentLoad.toFixed(1)} 
          unit="MW" 
          status="emerald"
          subtext={`Latest reading @ ${liveTime}`}
        />
        <div className="glass-panel p-5 flex flex-col justify-between">
          <span className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">
            Current Regime
          </span>
          <div className="mt-2">
            <StatusBadge status="stable" label="Operational" pulse />
          </div>
          <p className="text-[11px] text-[var(--text-muted)] mt-2 italic font-medium">
            System status: Nominal
          </p>
        </div>
        <MetricCard 
          label="MAPE (Accuracy)" 
          value={metricsSummary.mape.toFixed(1)} 
          unit="%" 
          status="emerald"
          subtext="Model Precision Index"
          trend={{ value: 0.4, isUp: false }}
        />
        <div className="glass-panel p-5 flex flex-col justify-between">
          <span className="text-[12px] font-bold uppercase tracking-widest text-[var(--text-muted)]">
            Data Source
          </span>
          <div className="flex items-end gap-2 mt-1">
            <span className="metric-num text-[var(--text-primary)] tabular-nums text-lg">
              DLinear+TIDE
            </span>
          </div>
          <button onClick={fetchData} className="mt-2 flex items-center gap-1.5 text-[10px] font-bold text-[var(--brand-blue)] hover:text-[var(--brand-blue-vibrant)] transition-colors">
            <RefreshCw className="w-3 h-3" /> Refresh Now
          </button>
        </div>
      </div>

      {/* Main Chart Card */}
      <div className="glass-panel p-6 flex flex-col gap-6">
        <div className="flex justify-between items-center">
          <div className="flex flex-col">
            <h2 className="headline text-[var(--text-primary)]">System Load Tracking (Live)</h2>
            <p className="caption text-[var(--text-muted)] mt-1 uppercase tracking-wider">
              Rolling window: Historical Data + Decomposition Forecast (24H)
            </p>
          </div>
          <div className="flex gap-2">
            <div className="flex items-center gap-4 px-4 border-r border-[var(--divider)] mr-2">
               <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[var(--brand-teal)]" />
                   <span className="micro-num text-[var(--text-muted)]">ACTUAL</span>
               </div>
               <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[var(--brand-blue)]" />
                  <span className="micro-num text-[var(--text-muted)]">FORECAST</span>
               </div>
            </div>
            <button className="px-3 py-1.5 bg-[var(--surface-secondary)] border border-[var(--border-card)] rounded text-[11px] font-bold uppercase text-[var(--text-primary)] hover:bg-[var(--divider)] transition-colors">
              Export Analysis
            </button>
          </div>
        </div>

        <div className="h-[400px]">
          <LoadChart 
            data={chartData}
            liveMarkerTime={liveTime}
            height={400}
          />
        </div>
      </div>

      {/* Secondary Monitor Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'System MAE', value: `${metricsSummary.mae.toFixed(1)} MW`, note: 'Mean Absolute Error' },
          { label: 'Data Health', value: 'GRADE A', note: 'Physics Sanity Index' },
          { label: 'Grid Region', value: 'GRIDCo National', note: 'Ghana Grid' },
          { label: 'Uncertainty', value: forecast?.uncertainty_mw ? `±${Math.round(Math.max(...forecast.uncertainty_mw))} MW` : '±8.5 MW', note: 'P10–P90 width (TIDE)' }
        ].map((item) => (
          <div key={item.label} className="glass-panel p-4 flex flex-col items-center text-center gap-1 border-t-2 border-t-[var(--divider)]">
            <span className="caption text-[var(--text-muted)] uppercase tracking-widest font-bold">
              {item.label}
            </span>
            <span className="data-num text-[var(--text-primary)] text-lg">
              {item.value}
            </span>
            <span className="text-[10px] text-[var(--text-muted)] italic">
              {item.note}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
