'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Area, ComposedChart, Bar,
} from 'recharts';

import {
  BarChart3, TrendingUp, Calendar, Activity,
  Loader2, AlertTriangle, RefreshCw, Zap, Clock, Upload, X, CheckCircle2, ChevronDown,
} from 'lucide-react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { dispatchForecastService, DispatchForecastResponse } from '@/services/dispatchForecastService';
import { baselineForecastService, BaselineForecastResponse, DataFreshnessInfo } from '@/services/baselineForecastService';
import { Forecast7DayView } from './Forecast7DayView';
import { Forecast30DayView } from './Forecast30DayView';
import { Forecast90DayView } from './Forecast90DayView';
import { ModelAccuracyPanel } from './ModelAccuracyPanel';

type ForecastTab = '24h' | '7d' | '30d' | '90d';
type EngineType = 'decom' | 'baseline';

function formatHour(h: number) {
  return ${h.toString().padStart(2, '0')}:00;
}

function Forecast24hView({ engineType }: { engineType: EngineType }) {
  const [forecast, setForecast] = useState<DispatchForecastResponse | BaselineForecastResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showFactors, setShowFactors] = useState(false);

  const fetchForecast = async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const data = engineType === 'decom'
        ? await dispatchForecastService.getTomorrow(forceRefresh)
        : await baselineForecastService.getTomorrow(forceRefresh);
      setForecast(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchForecast(); }, [engineType]);

  const chartData = useMemo(() => {
    if (!forecast) return [];
    if (engineType === 'baseline') {
      const bf = forecast as BaselineForecastResponse;
      return Array.from({ length: 24 }, (_, i) => ({
        hour: formatHour(i + 1),
        forecast: Math.round(bf.forecast_mw[i]),
        level_mw: bf.factors?.level_mw ?? 0,
        profile_pct: bf.factors?.profile ? parseFloat((bf.factors.profile[i] * 100).toFixed(1)) : 0,
        dow_offset: bf.factors?.dow_offset ?? 0,
      }));
    }
      const df = forecast as DispatchForecastResponse;
    return Array.from({ length: 24 }, (_, i) => ({
      hour: formatHour(i + 1),
      forecast: Math.round(df.forecast_mw[i]),
      p10: df.p10_mw ? Math.round(df.p10_mw[i]) : Math.round(df.forecast_mw[i] * 0.93),
      p90: df.p90_mw ? Math.round(df.p90_mw[i]) : Math.round(df.forecast_mw[i] * 1.07),
      band: [df.p10_mw ? Math.round(df.p10_mw[i]) : Math.round(df.forecast_mw[i] * 0.93), df.p90_mw ? Math.round(df.p90_mw[i]) : Math.round(df.forecast_mw[i] * 1.07)],
      uncertainty: df.uncertainty_mw ? Math.round(df.uncertainty_mw[i]) : 0,
      temperature: df.temperature_c ? df.temperature_c[i] : null,
      trend: df.components ? Math.round(df.components.trend[i]) : 0,
      temp_effect: df.components ? Math.round(df.components.temp_effect[i]) : 0,
      growth_effect: df.components ? Math.round(df.components.growth_effect[i]) : 0,
      kalman_bias: df.components ? Math.round(df.components.kalman_bias[i]) : 0,
      seasonal_ratio: df.factors ? parseFloat((df.factors.seasonal_ratio[i] * 100).toFixed(1)) : 0,
      temp_ratio: df.factors ? parseFloat((df.factors.temp_ratio[i] * 100).toFixed(1)) : 0,
      growth_ratio: df.factors ? parseFloat(((df.factors.growth_ratio[i] - 1) * 100).toFixed(2)) : 0,
    }));
  }, [forecast, engineType]);

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[var(--brand-blue)] animate-spin" /></div>;
  }

  if (error) {
    return <div className="flex flex-col items-center justify-center py-20 gap-4">
      <AlertTriangle className="w-10 h-10 text-[var(--status-crimson)]" />
      <p className="text-sm text-[var(--text-secondary)]">{error}</p>
      <button onClick={() => fetchForecast(true)} className="px-4 py-2 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors"><RefreshCw className="w-3.5 h-3.5 inline mr-1.5" /> Refresh</button>
    </div>;
  }

  if (!forecast) return null;

  const peakDemand = Math.round(Math.max(...forecast.forecast_mw));
  const peakHour = forecast.forecast_mw.indexOf(Math.max(...forecast.forecast_mw)) + 1;
  const avgDemand = Math.round(forecast.forecast_mw.reduce((a, b) => a + b, 0) / 24);
  const minDemand = Math.round(Math.min(...forecast.forecast_mw));
  const totalEnergy = Math.round(forecast.forecast_mw.reduce((a, b) => a + b, 0));

  const isBaseline = engineType === 'baseline';

  return (<>
    {/* Summary Cards */}
    <div className="grid grid-cols-4 gap-3">
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
        <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Peak Demand</p>
        <p className="text-2xl font-bold text-[var(--status-amber)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{peakDemand.toLocaleString()} MW</p>
        <p className="text-[11px] text-[var(--text-muted)] mt-0.5">Hour {peakHour} ({formatHour(peakHour)})</p>
      </div>
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
        <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Average Demand</p>
        <p className="text-2xl font-bold text-[var(--brand-blue)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{avgDemand.toLocaleString()} MW</p>
        <p className="text-[11px] text-[var(--text-muted)] mt-0.5">24h mean</p>
      </div>
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
        <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Minimum Demand</p>
        <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{minDemand.toLocaleString()} MW</p>
        <p className="text-[11px] text-[var(--text-muted)] mt-0.5">Off-peak</p>
      </div>
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
        <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Total Energy</p>
        <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{totalEnergy.toLocaleString()} MWh</p>
        <p className="text-[11px] text-[var(--text-muted)] mt-0.5">24h total</p>
      </div>
    </div>

    {/* Main Chart - always visible */}
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">24-Hour ECG Demand Forecast {isBaseline ? '(Baseline WT+DOW)' : '(DLinear+TIDE)'}</h3>
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="hour" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
          <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={['auto', 'auto']} />
          <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {!isBaseline && (
            <Area type="monotone" dataKey="band" stroke="none" fill="#4B6FFF" fillOpacity={0.35} legendType="none" />
          )}
          <Line type="monotone" dataKey="forecast" stroke="#4B6FFF" strokeWidth={2.5} dot={false} name="ECG Forecast" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>

    {/* Temperature Chart */}
    {!isBaseline && (chartData[0] as any).temperature != null && (
      <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-[0.8px]">
            Forecast Temperature &mdash; Accra, Ghana
          </h4>
          <span className="text-xs text-[var(--text-muted)]">
            {(chartData as any[]).filter((d: any) => d.temperature != null).length > 0
              ? ${Math.min(...(chartData as any[]).map((d: any) => d.temperature).filter((t: any) => t != null))}&deg;C &ndash;&deg;C
              : 'No temp data'}
          </span>
        </div>
        <ResponsiveContainer width="100%" height={100}>
          <ComposedChart data={chartData as any[]}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
            <XAxis dataKey="hour" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={['auto', 'auto']} width={30} />
            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 11 }}
              formatter={(v: number) => [${v.toFixed(1)}&deg;C, undefined]} />
              <Area type="monotone" dataKey="temperature" stroke="none" fill="#FF6B35" fillOpacity={0.25} legendType="none" />
            <Line type="monotone" dataKey="temperature" stroke="#FF6B35" strokeWidth={2} dot={false} name="Temperature &deg;C" />


          </ComposedChart>
        </ResponsiveContainer>
      </div>
    )}

    {/* Collapsible Factors Section */}
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)]">
      <button
        onClick={() => setShowFactors(!showFactors)}
        className="w-full flex items-center justify-between px-5 py-3 hover:bg-[var(--surface-secondary)] transition-colors"
      >
        <span className="text-xs font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-[var(--brand-blue)]" />
          Forecast Factors & Composition
        </span>
        <ChevronDown className={w-4 h-4 text-[var(--text-muted)] transition-transform } />
      </button>

      {showFactors && (
        <div className="border-t border-[var(--border-card)]">
          {/* Baseline Factors */}
          {isBaseline && forecast.factors && (
            <div className="px-5 py-4">
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-[var(--surface-primary)] border border-[var(--border-card)] px-4 py-3 text-center">
                  <p className="text-[11px] uppercase text-[var(--text-muted)] font-semibold">Level</p>
                  <p className="text-lg font-bold text-[var(--brand-blue)]">{(forecast as BaselineForecastResponse).factors?.level_mw?.toLocaleString()} MW</p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">0.65&middot;L1 + 0.35&middot;L7</p>
                </div>
                <div className="bg-[var(--surface-primary)] border border-[var(--border-card)] px-4 py-3 text-center">
                  <p className="text-[11px] uppercase text-[var(--text-muted)] font-semibold">DOW Offset</p>
                  <p className="text-lg font-bold text-[var(--status-amber)]">{(forecast as BaselineForecastResponse).factors?.dow_offset ?? 0} MW</p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Today&apos;s DOW adjustment</p>
                </div>
                <div className="bg-[var(--surface-primary)] border border-[var(--border-card)] px-4 py-3 text-center">
                  <p className="text-[11px] uppercase text-[var(--text-muted)] font-semibold">Profile Range</p>
                  <p className="text-lg font-bold text-[var(--status-emerald)]">
                    {(() => { const p = (forecast as BaselineForecastResponse).factors?.profile; if (!p) return '-'; const vals = p.map(v => v * 100 - 100); return ${Math.min(...vals).toFixed(1)}% / +%; })()}
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Min / Max deviation from level</p>
                </div>
              </div>
              <h4 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Forecast = Level &times; Profile + DOW Offset</h4>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                  <XAxis dataKey="hour" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <YAxis yAxisId="mw" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} orientation="left" />
                  <YAxis yAxisId="pct" orientation="right" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={[80, 120]} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar yAxisId="pct" dataKey="profile_pct" fill="var(--status-emerald)" name="Profile (% of Level)" opacity={0.6} radius={[2, 2, 0, 0]} />
                  <Line yAxisId="mw" type="monotone" dataKey="level_mw" stroke="var(--text-muted)" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Base Level (MW)" />
                  <Line yAxisId="mw" type="monotone" dataKey="forecast" stroke="var(--brand-blue)" strokeWidth={2} dot={false} name="Forecast (MW)" />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex gap-4 text-[10px] text-[var(--text-muted)] mt-2">
                <span><span className="inline-block w-3 h-[2px] bg-[var(--brand-blue)] mr-1" /> Forecast MW</span>
                <span><span className="inline-block w-3 h-[2px] bg-[var(--text-muted)] mr-1" style={{ borderTop: '1px dashed var(--text-muted)', height: 0 }} /> Base Level</span>
                <span><span className="inline-block w-3 h-2 bg-[var(--status-emerald)] mr-1" style={{ opacity: 0.6 }} /> Profile % of Level</span>
              </div>
            </div>
          )}

          {/* DLinear+TIDE Components */}
          {!isBaseline && 'components' in forecast && forecast.components && (
            <div className="px-5 py-4">
              <h4 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Component Breakdown</h4>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                  <XAxis dataKey="hour" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-card)', borderRadius: 0, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="temp_effect" stackId="1" fill="var(--status-amber)" stroke="var(--status-amber)" name="Temp Effect" />
                  <Area type="monotone" dataKey="growth_effect" stackId="1" fill="var(--status-emerald)" stroke="var(--status-emerald)" name="Growth Effect" />
                  <Area type="monotone" dataKey="kalman_bias" stackId="1" fill="var(--status-violet)" stroke="var(--status-violet)" name="Kalman Bias" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Factors Table (DLinear+TIDE only) */}
          {forecast.factors && !isBaseline && (
            <div className="px-5 py-4 border-t border-[var(--border-card)]">
              <h4 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Forecast Factors</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  <thead>
                    <tr>
                      <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Hour</th>
                      {Array.from({ length: 24 }, (_, i) => (
                        <th key={i} className="px-2 py-2 text-center text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">{formatHour(i + 1)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="px-3 py-2 text-xs font-medium text-[var(--text-primary)] border border-[var(--border-card)]">Seasonal Ratio (%)</td>
                      {Array.from({ length: 24 }, (_, i) => (
                        <td key={i} className="px-2 py-2 text-center border border-[var(--border-card)] text-[var(--text-secondary)]">{(forecast as DispatchForecastResponse).factors!.seasonal_ratio[i].toFixed(1)}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-3 py-2 text-xs font-medium text-[var(--text-primary)] border border-[var(--border-card)]">Temp Ratio (%)</td>
                      {Array.from({ length: 24 }, (_, i) => (
                        <td key={i} className="px-2 py-2 text-center border border-[var(--border-card)] text-[var(--text-secondary)]">{(forecast as DispatchForecastResponse).factors!.temp_ratio[i].toFixed(1)}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-3 py-2 text-xs font-medium text-[var(--text-primary)] border border-[var(--border-card)]">Growth Ratio (%)</td>
                      {Array.from({ length: 24 }, (_, i) => (
                        <td key={i} className="px-2 py-2 text-center border border-[var(--border-card)] text-[var(--text-muted)]">{((forecast as DispatchForecastResponse).factors!.growth_ratio[i] - 1).toFixed(2)}</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  </>);
}


export function ForecastView() {
  const [activeTab, setActiveTab] = useState<ForecastTab>('24h');
  const [engineType, setEngineType] = useState<EngineType>('decom');
  const [freshness, setFreshness] = useState<DataFreshnessInfo | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [engineHealth, setEngineHealth] = useState<{checkpoints_loaded?: number; mae_24h?: number; is_trained?: boolean} | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();

  useEffect(() => {
    baselineForecastService.getFreshness().then(setFreshness).catch(() => {});
    dispatchForecastService.getHealth().then((h) => {
      const dh = h?.engine_health?.dlinear_tide || {};
      setEngineHealth({ checkpoints_loaded: dh.checkpoints_loaded, mae_24h: dh.mae_24h, is_trained: dh.is_trained });
    }).catch(() => {});
  }, []);

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const res = await baselineForecastService.uploadScadaData(uploadFile);
      setUploadResult(Uploaded  days ( records). Latest: );
      setFreshness(res.freshness);
      setRefreshKey(k => k + 1);
      setTimeout(() => { setShowUpload(false); setUploadFile(null); setUploadResult(null); }, 2000);
    } catch (err: any) {
      setUploadError(err?.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const freshnessColor = freshness?.status === 'fresh' ? 'var(--status-emerald)'
    : freshness?.status === 'stale' ? 'var(--status-amber)'
    : freshness?.status === 'old' ? 'var(--status-crimson)'
    : 'var(--text-muted)';

  const navigateToDispatch = () => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('view', 'dispatch');
    router.replace(${pathname}?);
  };

  return (
    <div className="flex flex-col gap-4 max-w-[1440px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-[var(--bg-card)] border border-[var(--border-card)]">
        <div className="flex items-center gap-4">
          <h1 className="text-base font-semibold text-[var(--text-primary)]">
            <span className="text-[var(--brand-blue)]">GRIDCo</span> ECG Demand Forecast
          </h1>
          {freshness && (
            <div className="flex items-center gap-1.5 text-[11px]" style={{ color: freshnessColor }}>
              <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ backgroundColor: freshnessColor }} />
              <span className="font-semibold uppercase tracking-[0.5px]">Data</span>
              <span>{freshness.latest_date}</span>
              <span>&middot;</span>
              <span>{freshness.days_stale > 0 ? ${freshness.days_stale}d stale : 'current'}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => setShowUpload(true)} className="px-4 py-1.5 text-xs font-semibold border border-[var(--status-emerald)] text-[var(--status-emerald)] hover:bg-[var(--status-emerald)]/10 transition-colors flex items-center gap-1.5">
            <Upload className="w-3.5 h-3.5" /> Upload SCADA
          </button>
          <button onClick={navigateToDispatch} className="px-4 py-1.5 text-xs font-semibold border border-[var(--brand-blue)] text-[var(--brand-blue)] hover:bg-[var(--brand-blue)]/10 transition-colors flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" /> Apply to Schedule
          </button>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[var(--bg-card)] border border-[var(--border-card)] w-full max-w-lg mx-4">
            <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border-card)]">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Upload SCADA Data</h2>
              <button onClick={() => { setShowUpload(false); setUploadFile(null); setUploadResult(null); setUploadError(null); }}>
                <X className="w-4 h-4 text-[var(--text-muted)] hover:text-[var(--text-primary)]" />
              </button>
            </div>
            <div className="px-5 py-4 space-y-4">
              <p className="text-xs text-[var(--text-secondary)]">
                Upload a CSV file with recent SCADA demand data (columns: date, hour, demand_mw).
                Each day must have 24 complete hourly records. This will update the forecast level immediately.
              </p>
              <div className="border-2 border-dashed border-[var(--border-card)] p-6 text-center">
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  id="scada-upload"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
                <label htmlFor="scada-upload" className="cursor-pointer flex flex-col items-center gap-2">
                  <Upload className="w-8 h-8 text-[var(--text-muted)]" />
                  <span className="text-sm text-[var(--text-muted)]">
                    {uploadFile ? uploadFile.name : 'Click to select CSV file'}
                  </span>
                  {uploadFile && (
                    <span className="text-xs text-[var(--text-secondary)]">
                      {(uploadFile.size / 1024).toFixed(1)} KB
                    </span>
                  )}
                </label>
              </div>
              {uploadError && (
                <div className="flex items-center gap-2 text-xs text-[var(--status-crimson)] bg-[var(--status-crimson)]/5 px-3 py-2 border border-[var(--status-crimson)]/20">
                  <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                  {uploadError}
                </div>
              )}
              {uploadResult && (
                <div className="flex items-center gap-2 text-xs text-[var(--status-emerald)] bg-[var(--status-emerald)]/5 px-3 py-2 border border-[var(--status-emerald)]/20">
                  <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
                  {uploadResult}
                </div>
              )}
            </div>
            <div className="flex justify-end gap-2 px-5 py-3 border-t border-[var(--border-card)]">
              <button
                onClick={() => { setShowUpload(false); setUploadFile(null); setUploadResult(null); setUploadError(null); }}
                className="px-4 py-1.5 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={!uploadFile || uploading}
                className="px-4 py-1.5 text-xs font-semibold bg-[var(--status-emerald)] text-white hover:bg-[var(--status-emerald)]/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                {uploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
                {uploading ? 'Uploading...' : 'Upload & Refit'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Engine Toggle + Info Bar */}
      <div className="flex justify-between items-center px-4 py-2.5 bg-[var(--bg-card)] border border-[var(--border-card)] text-xs">
        <div className="flex items-center gap-3">
          <span className="text-[var(--text-muted)]">Engine:</span>
          <div className="flex border border-[var(--border-card)]">
            <button onClick={() => setEngineType('decom')} className={px-3 py-1 text-xs font-semibold transition-colors }>
              DLinear+TIDE
            </button>
            <button onClick={() => setEngineType('baseline')} className={px-3 py-1 text-xs font-semibold transition-colors }>
              Baseline WT+DOW
            </button>
          </div>
        </div>
        <div>
          <span className="text-[var(--text-muted)]">MAE:</span>
          <span className="text-[var(--text-primary)] font-semibold ml-1">
            {engineType === 'baseline'
              ? '98 MW'
              : (engineHealth?.mae_24h ? ${Math.round(engineHealth.mae_24h)} MW : '67 MW')}
          </span>
          {engineType === 'decom' && engineHealth?.checkpoints_loaded && (
            <span className="text-[10px] text-[var(--text-muted)] ml-2">
              ({engineHealth.checkpoints_loaded} ckpts{engineHealth.is_trained ? '·corrector ok' : ''})
            </span>
          )}
        </div>
        <div><span className="text-[var(--text-muted)]">Horizons:</span> <span className="text-[var(--text-primary)] font-semibold ml-1">1d-7d hourly / 30d daily / 90d weekly</span></div>
      </div>

      {/* Horizon Tabs */}
      <div className="flex border border-[var(--border-card)]">
        <button onClick={() => setActiveTab('24h')} className={px-5 py-2 text-xs font-semibold transition-colors }>
          <Clock className="w-3.5 h-3.5 inline mr-1.5" /> 24 Hours
        </button>
        <button onClick={() => setActiveTab('7d')} className={px-5 py-2 text-xs font-semibold transition-colors }>
          <BarChart3 className="w-3.5 h-3.5 inline mr-1.5" /> 7 Days
        </button>
        <button onClick={() => setActiveTab('30d')} className={px-5 py-2 text-xs font-semibold transition-colors }>
          <Calendar className="w-3.5 h-3.5 inline mr-1.5" /> 30 Days
        </button>
        <button onClick={() => setActiveTab('90d')} className={px-5 py-2 text-xs font-semibold transition-colors }>
          <TrendingUp className="w-3.5 h-3.5 inline mr-1.5" /> 90 Days
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === '24h' && <Forecast24hView key={24h--} engineType={engineType} />}
      {activeTab === '7d' && <Forecast7DayView key={7d--} engineType={engineType} />}
      {activeTab === '30d' && <Forecast30DayView key={30d--} engineType={engineType} />}
      {activeTab === '90d' && <Forecast90DayView key={90d--} engineType={engineType} />}

      {/* Model Accuracy & Error Insights */}
      <ModelAccuracyPanel horizon={activeTab === '24h' ? '24h' : activeTab === '7d' ? '7d' : activeTab === '30d' ? '30d' : '90d'} />

      {/* Footer */}
      <div className="flex justify-between text-[11px] text-[var(--text-muted)] px-1 py-2">
        <span>Forecast generated by {engineType === 'baseline' ? 'WeightedTrendEngine (0.65&middot;L1 + 0.35&middot;L7 + DOW offset &times; Month&times;DOW profile)' : 'DLinear+TIDE &mdash; 6-fold ensemble + TIDE bias corrector + per-hour uncertainty'}</span>
      </div>
    </div>
  );
}