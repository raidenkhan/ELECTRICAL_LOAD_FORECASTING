'use client';

import {
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Info,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ForecastChart } from './ForecastChart';
import { ActiveAlerts } from './ActiveAlerts';
import { useState, useEffect } from 'react';
import { ExplainabilityModal } from './ExplainabilityModal';
import { forecastService, ForecastResponse } from '@/services/forecastService';


export function OverviewDashboard() {
  const [showExplainability, setShowExplainability] = useState(false);
  const [stlf, setStlf] = useState<ForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        const data = await forecastService.getSTLF(24);
        setStlf(data);
      } catch (err: any) {
        console.error('Failed to fetch STLF:', err);
        setError(err.message || 'Failed to load forecast data');
      } finally {
        setIsLoading(false);
      }
    }
    fetchData();
  }, []);

  const chartData = stlf && stlf.forecast_mw?.length > 0 ? stlf.timestamps.map((t, i) => ({
    time: new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    forecast: Math.round(stlf.forecast_mw[i] * 10) / 10,
    upper: stlf.p90 ? Math.round(stlf.p90[i] * 10) / 10 : (stlf.forecast_mw[i] * 1.1),
    lower: stlf.p10 ? Math.round(stlf.p10[i] * 10) / 10 : (stlf.forecast_mw[i] * 0.9),
    peak: stlf.forecast_mw[i] === Math.max(...stlf.forecast_mw)
  })) : [];

  const regimeData = stlf?.regime_distribution ?? [];

  const currentLoad = stlf && stlf.forecast_mw?.length > 0 ? Math.round(stlf.forecast_mw[0]) : 1247;
  const peakLoad = stlf && stlf.forecast_mw?.length > 0 ? Math.round(Math.max(...stlf.forecast_mw)) : 1580;
  const peakTime = stlf && stlf.forecast_mw?.length > 0
    ? new Date(stlf.timestamps[stlf.forecast_mw.indexOf(Math.max(...stlf.forecast_mw))]).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '14:30';

  return (
    <div className="space-y-12 p-12" style={{ backgroundColor: 'var(--bg-page)' }}>
      {error && (
        <div className="p-6 bg-red-950/20 border border-red-500/30 text-red-400 text-[13px] flex items-center gap-3 glass-morphism">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-bold tracking-tight uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>{error}</span>
        </div>
      )}

      {/* Primary Landing - Forecasting Overview */}
      <div className="flex flex-col gap-4">
        <h1 className="text-4xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
          Forecasting Overview
        </h1>
        <p className="text-sm font-medium opacity-40 uppercase tracking-[0.3em]" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
          State: Regional Electrical Grid / Node: NY-SOTA-01 / Real-time sync: Active
        </p>
      </div>

      {/* Key Metrics Grid - Refined V3 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">

        <MetricCard
          label="Estimated Load"
          value={isLoading ? "..." : currentLoad.toLocaleString()}
          unit="MW"
          change="+2.4%"
          changeType="up"
          isLive={true}
          subtitle="T+15m forecast"
        />
        <MetricCard
          label="Forecast Accuracy"
          value="96.8"
          unit="%"
          change="+0.3%"
          changeType="up"
          subtitle="Last 24 hours"
        />
        <MetricCard
          label="Peak Forecast"
          value={isLoading ? "..." : peakLoad.toLocaleString()}
          unit="MW"
          time={isLoading ? "..." : peakTime}
          subtitle="Today's expected peak"
        />
        <MetricCard
          label="System Status"
          value="Normal"
          status="operational"
          subtitle="All systems operational"
        />
      </div>

      {/* Forecast Chart - Refined V3 */}
      <ForecastChart data={chartData} isLoading={isLoading} />

      {/* Active Alerts - Refined V3 */}
      <ActiveAlerts />

      {/* Full Width Bottom Grid */}
      <div className="flex flex-col gap-12">

        {/* Regime Probability */}
        <div
          className="glass-morphism"
        >

          <div className="px-6 py-5 border-b" style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-1.5 h-4" style={{ backgroundColor: 'var(--lime-primary)' }} />
                <h3 className="text-[13px] font-bold tracking-[0.2em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
                  Operating Regime Distribution
                </h3>
              </div>
              <button
                className="p-2 hover:bg-white/5 transition-colors"
                title="More information"
                onClick={() => setShowExplainability(true)}
              >
                <Info className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
              </button>
            </div>
          </div>


          <div className="p-8">
            {isLoading ? (
              <div className="flex items-center justify-center h-[260px]">
                <span className="text-xs font-bold tracking-widest uppercase animate-pulse" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-geist-mono)' }}>COMPUTING REGIMES...</span>
              </div>
            ) : regimeData.length === 0 ? (
              <div className="flex items-center justify-center h-[260px]">
                <span className="text-xs font-bold tracking-widest uppercase" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-geist-mono)' }}>NO DATA</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={regimeData} barGap={8}>
                  <defs>
                    <linearGradient id="regimePeak" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--regime-peak)" stopOpacity={1} />
                      <stop offset="100%" stopColor="var(--regime-peak)" stopOpacity={0.6} />
                    </linearGradient>
                    <linearGradient id="regimeTransition" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--regime-transition)" stopOpacity={1} />
                      <stop offset="100%" stopColor="var(--regime-transition)" stopOpacity={0.6} />
                    </linearGradient>
                    <linearGradient id="regimeStandard" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--regime-standard)" stopOpacity={1} />
                      <stop offset="100%" stopColor="var(--regime-standard)" stopOpacity={0.6} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" strokeOpacity={0.1} vertical={false} />

                  <XAxis
                    dataKey="hour"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: '10px', fontWeight: 'bold', fontFamily: 'var(--font-geist-mono)' }}
                    tickLine={false}
                    axisLine={{ stroke: '#FFFFFF08' }}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: '10px', fontWeight: 'bold', fontFamily: 'var(--font-geist-mono)' }}
                    tickLine={false}
                    axisLine={{ stroke: '#FFFFFF08' }}
                    tickFormatter={(val) => `${val}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(24, 24, 27, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '0px',
                      fontSize: '11px',
                      fontFamily: 'var(--font-geist-mono)',
                      backdropFilter: 'blur(10px)',
                      boxShadow: 'var(--glass-shadow)'
                    }}
                    itemStyle={{ padding: '2px 0' }}
                    cursor={{ fill: 'rgba(255, 255, 255, 0.03)' }}
                  />
                  <Bar
                    dataKey="regime0"
                    stackId="a"
                    fill="url(#regimeStandard)"
                    name="Standard"
                    barSize={32}
                  />
                  <Bar
                    dataKey="regime1"
                    stackId="a"
                    fill="url(#regimeTransition)"
                    name="Transition"
                    barSize={32}
                  />
                  <Bar
                    dataKey="regime2"
                    stackId="a"
                    fill="url(#regimePeak)"
                    name="Peak"
                    barSize={32}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}



            {/* Legend Info */}
            <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t" style={{ borderColor: 'var(--border-primary)' }}>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3" style={{ backgroundColor: 'var(--regime-standard)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
                    STANDARD
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Normal load patterns
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3" style={{ backgroundColor: 'var(--regime-transition)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
                    TRANSITION
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Volatile conditions
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3" style={{ backgroundColor: 'var(--regime-peak)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
                    PEAK
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  High demand periods
                </p>
              </div>
            </div>

          </div>
        </div>

        {/* System Status */}
        <div
          className="glass-morphism"
        >
          <div className="px-6 py-5 border-b" style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}>
            <div className="flex items-center gap-3">
              <div className="w-1.5 h-4" style={{ backgroundColor: 'var(--lime-primary)' }} />
              <h3 className="text-[13px] font-bold tracking-[0.2em] uppercase" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
                System Health
              </h3>
            </div>
          </div>



          <div className="p-0 flex flex-col">
            <StatusItem
              label="DATA FEED"
              status="operational"
              detail="Last update: 2 minutes ago"
              metric="15-MIN INTERVALS"
            />
            <StatusItem
              label="MODEL ENGINE"
              status="operational"
              detail="LSTM Ensemble v2.3"
              metric="OK / ACTIVE"
            />
            <StatusItem
              label="API GATEWAY"
              status="operational"
              detail="VPC Endpoint Connected"
              metric="OK"
            />
          </div>


          {/* System Info Footer */}
          <div
            className="px-6 py-4 border-t"
            style={{
              borderColor: 'var(--border-primary)',
              backgroundColor: 'var(--hover-bg)'
            }}
          >


            <div className="flex items-center justify-between text-xs">
              <div>
                <span style={{ color: 'var(--text-muted)' }}>System Uptime:</span>
                <span className="ml-2 font-semibold" style={{ color: 'var(--text-primary)' }}>
                  99.8% (Last 30 days)
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Last Restart:</span>
                <span className="ml-2 font-semibold" style={{ color: 'var(--text-primary)' }}>
                  2 days ago
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Explainability Modal */}
      <ExplainabilityModal
        isOpen={showExplainability}
        onClose={() => setShowExplainability(false)}
      />
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  change?: string;
  changeType?: 'up' | 'down';
  isLive?: boolean;
  time?: string;
  status?: 'operational' | 'warning' | 'error';
  subtitle?: string;
}

function MetricCard({
  label,
  value,
  unit,
  change,
  changeType,
  isLive,
  time,
  status,
  subtitle
}: MetricCardProps) {
  const ChangeIcon = changeType === 'up' ? ArrowUpRight : ArrowDownRight;

  const statusConfig = {
    operational: { color: 'var(--lime-primary)', label: 'STABLE' },
    warning: { color: 'var(--status-warn)', label: 'WARNING' },
    error: { color: 'var(--status-error)', label: 'CRITICAL' }
  };

  return (
    <div
      className="px-8 py-10 glass-morphism relative overflow-hidden group transition-all duration-300"
    >

      <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden pointer-events-none opacity-0 group-hover:opacity-20 transition-opacity">
        <div className="absolute top-0 right-0 w-[1px] h-4 bg-white" />
        <div className="absolute top-0 right-0 w-4 h-[1px] bg-white" />
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3
              className="text-[10px] font-bold tracking-widest uppercase mb-1"
              style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)', fontSize: '10px' }}
            >
              {label}
            </h3>
            {isLive && (
              <div className="flex items-center gap-1.5">
                <div
                  className="w-1.5 h-1.5 animate-pulse"
                  style={{ backgroundColor: 'var(--lime-primary)', boxShadow: 'var(--lime-glow)' }}
                />
                <span className="text-[10px] font-bold" style={{ color: 'var(--lime-primary)', fontFamily: 'var(--font-geist-mono)', fontSize: '10px' }}>
                  LIVE
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Value */}
      <div className="mb-6">
        <div className="flex items-baseline gap-4">
          <span
            className="font-black tracking-tighter leading-none"
            style={{ color: 'var(--text-primary)', fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            {value}
          </span>
          {unit && (
            <span
              className="text-lg font-bold opacity-40"
              style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-geist-mono)' }}
            >
              {unit}
            </span>
          )}
          {time && (
            <span
              className="text-sm font-bold opacity-40"
              style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}
            >
              @ {time}
            </span>
          )}
          {status && (
            <span
              className="px-3 py-1.5 font-bold ml-auto border bg-white/[0.02] tracking-widest"
              style={{
                borderColor: statusConfig[status].color,
                color: statusConfig[status].color,
                fontFamily: 'var(--font-geist-mono)',
                fontSize: '11px'
              }}
            >
              {statusConfig[status].label}
            </span>
          )}
        </div>
      </div>


      {/* Footer */}
      <div className="flex items-center justify-between border-t border-white/5 pt-4">
        <span style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)', fontSize: '10px', fontWeight: 600 }}>
          {subtitle}
        </span>
        {change && (
          <div
            className="flex items-center gap-2 font-bold"
            style={{
              color: changeType === 'up' ? 'var(--lime-primary)' : 'var(--status-error)',
              fontSize: '11px'
            }}
          >
            <ChangeIcon className="w-3.5 h-3.5" />
            {change}
          </div>
        )}
      </div>
    </div>
  );
}


interface StatusItemProps {
  label: string;
  status: 'operational' | 'degraded' | 'offline';
  detail: string;
  metric?: string;
}

function StatusItem({ label, status, detail, metric }: StatusItemProps) {
  const statusConfig = {
    operational: { color: 'var(--lime-primary)', label: 'OK' },
    degraded: { color: 'var(--status-warn)', label: 'WARN' },
    offline: { color: 'var(--status-error)', label: 'OFFLINE' }
  };

  const config = statusConfig[status];

  return (
    <div
      className="flex items-center justify-between px-6 py-4 border-b transition-colors hover:bg-white/[0.02] group"
      style={{ borderColor: 'var(--border-primary)', opacity: 0.8 }}
    >

      <div className="flex items-center gap-4 flex-1">
        <div
          className="w-1.5 h-1.5 flex-shrink-0"
          style={{
            backgroundColor: config.color,
            boxShadow: `0 0 10px ${config.color}`,
            opacity: status === 'operational' ? 1 : 0.6
          }}
        />
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-bold tracking-wider uppercase mb-0.5" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-geist-mono)' }}>
            {label}
          </p>
          <p className="text-[10px] font-bold uppercase opacity-40 group-hover:opacity-60 transition-opacity" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
            {detail}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span
          className="px-2 py-0.5 text-[9px] font-bold border transition-colors"
          style={{
            borderColor: `${config.color}40`,
            color: config.color,
            fontFamily: 'var(--font-geist-mono)'
          }}
        >
          {config.label}
        </span>
        {metric && (
          <div className="text-[10px] font-bold text-right min-w-[100px] uppercase opacity-40 group-hover:opacity-60 transition-opacity" style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-geist-mono)' }}>
            {metric}
          </div>
        )}
      </div>
    </div>
  );
}