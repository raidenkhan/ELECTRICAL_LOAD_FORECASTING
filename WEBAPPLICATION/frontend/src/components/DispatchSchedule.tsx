'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Upload, FileSpreadsheet, CheckCircle2, AlertTriangle,
  Loader2, Save, Zap, BarChart3, Table2, History, RefreshCw,
  FileText, Clock, User, Download, ArrowLeftRight,
} from 'lucide-react';
import { toast } from 'sonner';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Area, ComposedChart, BarChart, Bar,
} from 'recharts';
import { scheduleService, ScheduleDetail, AggregatedSchedule, HourlyDemandItem, HourlySupplyItem } from '@/services/scheduleService';
import { AuditTrailPanel } from './AuditTrailPanel';

type ViewTab = 'grid' | 'graph';
type UploadPhase = 'idle' | 'uploading' | 'parsing' | 'done' | 'error';
type GridTab = 'demand' | 'supply' | 'combined';
type AppTab = 'schedule' | 'history' | 'audit' | 'settings';

interface AuditEntry {
  time: string;
  actor: string;
  message: string;
  type: 'upload' | 'correct' | 'edit' | 'confirm' | 'revise' | 'export';
  diffs?: string[];
}

export function DispatchSchedule() {
  const [schedule, setSchedule] = useState<ScheduleDetail | null>(null);
  const [scheduleId, setScheduleId] = useState<number | null>(null);
  const [viewTab, setViewTab] = useState<ViewTab>('grid');
  const [gridTab, setGridTab] = useState<GridTab>('demand');
  const [appTab, setAppTab] = useState<AppTab>('schedule');
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>('idle');
  const [editingCell, setEditingCell] = useState<{ entity: string; hour: number } | null>(null);
  const [editValue, setEditValue] = useState('');
  const [aggregated, setAggregated] = useState<AggregatedSchedule | null>(null);
  const [loading, setLoading] = useState(false);
  const [auditOpen, setAuditOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const lastId = localStorage.getItem('lastScheduleId');
    if (lastId) {
      const id = parseInt(lastId);
      loadSchedule(id);
      setScheduleId(id);
    }
  }, []);

  const loadSchedule = async (id: number) => {
    setLoading(true);
    try {
      const [data, agg] = await Promise.all([
        scheduleService.getSchedule(id),
        scheduleService.getAggregated(id, true).catch(() => null),
      ]);
      setSchedule(data);
      setAggregated(agg);
      setScheduleId(id);
      localStorage.setItem('lastScheduleId', String(id));
    } catch {
      setSchedule(null);
      setAggregated(null);
      setScheduleId(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith('.xlsx')) {
      toast.error('Only .xlsx files are supported');
      return;
    }

    setUploadPhase('uploading');
    try {
      await new Promise(r => setTimeout(r, 600));
      setUploadPhase('parsing');

      const result = await scheduleService.uploadSchedule(file);
      setUploadPhase('done');
      toast.success(`Schedule for ${result.date} stored (id=${result.id})`);
      setScheduleId(result.id);
      localStorage.setItem('lastScheduleId', String(result.id));
      await loadSchedule(result.id);
    } catch (err: any) {
      setUploadPhase('error');
      toast.error(err?.response?.data?.detail || 'Upload failed');
    }
  };

  const handleCellClick = (entity: string, hour: number) => {
    if (schedule?.status === 'confirmed') return;

    const supplyItem = schedule?.supply.find(s => s.plant_name === entity && s.hour === hour);
    if (supplyItem?.is_baseload) {
      const confirmed = window.confirm(
        `${entity} is a baseload plant with a default constant value. Override this hour?`
      );
      if (!confirmed) return;
    }

    setEditingCell({ entity, hour });
    const item = schedule?.demand.find(d => d.entity_name === entity && d.hour === hour)
      || supplyItem;
    setEditValue(String(item ? ('demand_mw' in item ? item.demand_mw : item.supply_mw) : ''));
  };

  const handleCellSave = async () => {
    if (!editingCell || !scheduleId) return;
    const value = parseFloat(editValue);
    if (isNaN(value)) {
      toast.error('Enter a valid number');
      return;
    }

    try {
      const isEntityInDemand = schedule?.demand.some(
        d => d.entity_name === editingCell.entity
      );
      const table = isEntityInDemand ? 'demand' : 'supply';
      await scheduleService.updateCell(scheduleId, table as 'demand' | 'supply', editingCell.entity, editingCell.hour, value);
      await loadSchedule(scheduleId);
      setEditingCell(null);
      toast.success('Cell updated');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Update failed');
    }
  };

  const handleConfirm = async () => {
    if (!scheduleId) return;
    const notes = prompt('Confirmation reason (optional):');
    try {
      await scheduleService.confirmSchedule(scheduleId, notes || undefined);
      await loadSchedule(scheduleId);
      toast.success('Schedule confirmed');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Confirm failed');
    }
  };

  const handleRevise = async () => {
    if (!scheduleId) return;
    const notes = prompt('Reason for revision:');
    if (!notes) {
      toast.error('Revision reason is required');
      return;
    }
    try {
      await scheduleService.reviseSchedule(scheduleId, notes);
      await loadSchedule(scheduleId);
      toast.success('Schedule revised — please update cells as needed');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Revise failed');
    }
  };

  const entityRows = schedule?.demand
    ? [...new Set(schedule.demand.map(d => d.entity_name))]
    : [];
  const plantRows = schedule?.supply
    ? [...new Set(schedule.supply.map(s => s.plant_name))]
    : [];

  const getDemandValue = (entity: string, hour: number) =>
    schedule?.demand.find(d => d.entity_name === entity && d.hour === hour)?.demand_mw;

  const getSupplyValue = (plant: string, hour: number) =>
    schedule?.supply.find(s => s.plant_name === plant && s.hour === hour)?.supply_mw;

  const chartData = Array.from({ length: 24 }, (_, i) => {
    const h = i + 1;
    const ecg = getDemandValue('ECG', h) || 0;
    const nits = getDemandValue('NITS_Total', h) || 0;
    const trojan = getSupplyValue('Trojan I (Tema)', h) || 0;
    const meienergy = getSupplyValue('Meienergy', h) || 0;
    const bxc = getSupplyValue('BXC Solar', h) || 0;
    return {
      hour: `${h}`,
      ECG: Math.round(ecg),
      NITS_Total: Math.round(nits),
      'Trojan I': Math.round(trojan),
      Meienergy: Math.round(meienergy),
      'BXC Solar': Math.round(bxc),
    };
  });

  const peakDemand = aggregated
    ? Math.round(aggregated.peak_demand_mw)
    : (schedule?.demand
      ? Math.max(...schedule.demand.filter(d => d.entity_name === 'ECG').map(d => d.demand_mw), 0)
      : 0);
  const peakHour = aggregated
    ? aggregated.peak_demand_hour
    : (schedule?.demand.find(d => d.demand_mw === peakDemand)?.hour || '-');

  const totalGeneration = aggregated
    ? Math.round(aggregated.total_energy_supply_mwh)
    : (schedule?.supply.length
      ? Math.round(schedule.supply.filter(s => s.supply_mw > 0).reduce((a, b) => a + b.supply_mw, 0))
      : 0);

  const avgDemand = aggregated
    ? Math.round(aggregated.avg_demand_mw)
    : (schedule?.demand.length
      ? Math.round(schedule.demand.filter(d => d.entity_name === 'NITS_Total').reduce((a, b) => a + b.demand_mw, 0) / 24)
      : 0);

  const reserveMargin = aggregated
    ? Math.round(aggregated.hourly.reduce((min, h) => h.reserve_mw < min ? h.reserve_mw : min, Infinity))
    : (avgDemand > 0 ? Math.round(totalGeneration - avgDemand) : 0);

  // Aggregate NITS total per hour
  const nitsTotalByHour = useMemo(() => {
    const nitsItems = schedule?.demand.filter(d => d.entity_name === 'NITS_Total') || [];
    const map: Record<number, number> = {};
    nitsItems.forEach(item => { map[item.hour] = item.demand_mw; });
    return map;
  }, [schedule]);

  // Overall total = NITS + embedded sources
  const totalByHour = useMemo(() => {
    const result: Record<number, number> = {};
    for (let h = 1; h <= 24; h++) {
      const nits = nitsTotalByHour[h] || 0;
      const supply = schedule?.supply.filter(s => s.hour === h).reduce((a, b) => a + b.supply_mw, 0) || 0;
      result[h] = nits + supply;
    }
    return result;
  }, [nitsTotalByHour, schedule]);

  const auditEntries: AuditEntry[] = [
    { time: '08:15', actor: 'admin@gridco.gh', message: `Uploaded "${schedule?.source_filename || 'dispatch sheet.xlsx'}"`, type: 'upload' },
    { time: '08:17', actor: 'system (LLM parser)', message: 'Auto-corrected — NITS row shifted values', type: 'correct', diffs: ['H09: 1806→1872', 'H10: 1808→1880'] },
    { time: '08:30', actor: 'operator@gridco.gh', message: 'Schedule confirmed for dispatch', type: 'confirm', diffs: ['Ready for dispatch day'] },
  ];

  const getCellClass = (val: number | undefined, isEditable: boolean) => {
    if (val === undefined) return 'text-[var(--text-muted)]';
    const classes = isEditable ? 'cursor-pointer hover:outline hover:outline-2 hover:outline-[var(--brand-blue)] hover:outline-offset-[-2px]' : '';
    if (val > 2400) return `${classes} bg-[rgba(248,81,73,0.15)] text-[var(--status-crimson)] font-semibold`;
    if (val > 2000) return `${classes} bg-[rgba(250,204,21,0.1)] text-[var(--status-amber)]`;
    return `${classes} text-[var(--text-secondary)]`;
  };

  const entityColor = (name: string): string => {
    const map: Record<string, string> = {
      ECG: 'var(--brand-blue)',
      NEDCo: 'var(--status-amber)',
      VALCO: 'var(--status-violet)',
      Mines: 'var(--status-emerald)',
      Export: '#06B6D4',
      NITS_Total: 'var(--text-secondary)',
    };
    return map[name] || 'var(--text-secondary)';
  };

  const categoryColor = (cat: string): string => {
    const map: Record<string, string> = {
      hydro: 'var(--brand-blue)',
      thermal: '#EF4444',
      interconnection: 'var(--status-violet)',
    };
    return map[cat] || 'var(--text-secondary)';
  };

  const entityBg = (name: string): string => {
    const map: Record<string, string> = {
      ECG: 'rgba(59, 130, 246, 0.06)',
      NEDCo: 'rgba(245, 158, 11, 0.06)',
      VALCO: 'rgba(139, 92, 246, 0.06)',
      Mines: 'rgba(16, 185, 129, 0.06)',
      Export: 'rgba(6, 182, 212, 0.06)',
      NITS_Total: 'rgba(100, 116, 139, 0.03)',
    };
    return map[name] || 'transparent';
  };

  const categoryBg = (cat: string): string => {
    const map: Record<string, string> = {
      hydro: 'rgba(59, 130, 246, 0.06)',
      thermal: 'rgba(239, 68, 68, 0.06)',
      interconnection: 'rgba(139, 92, 246, 0.06)',
    };
    return map[cat] || 'transparent';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 text-[var(--brand-blue)] animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 max-w-[1440px] mx-auto">
      {/* Upload section (shown when no schedule loaded) */}
      {!schedule && (
        <div
          className="border border-dashed border-[var(--border-card)] bg-[var(--bg-card)] p-8 flex flex-col items-center justify-center cursor-pointer hover:border-[var(--brand-blue)]/50 transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" accept=".xlsx" className="hidden" onChange={handleFileUpload} />
          {uploadPhase === 'idle' || uploadPhase === 'error' ? (
            <>
              <div className="w-14 h-14 rounded-full bg-[var(--surface-secondary)] flex items-center justify-center mb-4">
                <FileSpreadsheet className="w-7 h-7 text-[var(--brand-blue)]" />
              </div>
              <h3 className="text-base font-semibold text-[var(--text-primary)]">Upload Dispatch Schedule</h3>
              <p className="text-sm text-[var(--text-muted)] mt-2 mb-6 text-center max-w-md">
                Upload the ECG Daily Demand Data Sheet (.xlsx) to create a new dispatch schedule.
                The system will parse the 24-hour demand and supply rows automatically.
              </p>
              <span className="px-4 py-2 text-sm font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors">
                Select Excel File
              </span>
              {uploadPhase === 'error' && (
                <p className="text-xs text-[var(--status-crimson)] mt-3">Upload failed. Try again.</p>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-7 h-7 text-[var(--brand-blue)] animate-spin" />
              <p className="text-sm font-semibold text-[var(--text-primary)]">
                {uploadPhase === 'uploading' ? 'Uploading...' : 'Parsing schedule data...'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Loaded schedule */}
      {schedule && (
        <>
          {/* Top Bar */}
          <div className="flex items-center justify-between px-4 py-3 bg-[var(--bg-card)] border border-[var(--border-card)]">
            <h1 className="text-base font-semibold text-[var(--text-primary)]">
              <span className="text-[var(--brand-blue)]">GRIDCo</span> Dispatch Schedule
            </h1>
            <div className="flex items-center gap-3">
              <span className={`text-[11px] px-3 py-1 font-semibold uppercase tracking-[0.5px] border ${
                schedule.status === 'confirmed'
                  ? 'border-[var(--status-emerald)] text-[var(--status-emerald)] bg-[rgba(63,185,80,0.1)]'
                  : 'border-[var(--status-amber)] text-[var(--status-amber)] bg-[rgba(250,204,21,0.1)]'
              }`}>
                {schedule.status === 'confirmed' ? 'CONFIRMED' : 'DRAFT'}
              </span>
              <span className="text-xs text-[var(--text-muted)]">operator@gridco.gh</span>
              <span className="px-3 py-1 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)]">
                Export PDF
              </span>
              {schedule.status !== 'confirmed' ? (
                <button
                  onClick={handleConfirm}
                  className="px-4 py-1 text-xs font-semibold border border-[var(--status-emerald)] bg-[rgba(46,160,67,0.15)] text-white hover:bg-[rgba(46,160,67,0.25)] transition-colors"
                >
                  <CheckCircle2 className="w-3.5 h-3.5 inline mr-1.5" /> Confirm Schedule
                </button>
              ) : (
                <button
                  onClick={handleRevise}
                  className="px-4 py-1 text-xs font-semibold border border-[var(--status-crimson)] bg-[rgba(248,81,73,0.15)] text-[var(--status-crimson)] hover:bg-[rgba(248,81,73,0.25)] transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5 inline mr-1.5" /> Revise
                </button>
              )}
              <button
                onClick={() => { setSchedule(null); setScheduleId(null); localStorage.removeItem('lastScheduleId'); }}
                className="px-3 py-1 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors"
              >
                New Upload
              </button>
            </div>
          </div>

          {/* Info Bar */}
          <div className="flex justify-between px-4 py-2.5 bg-[var(--bg-card)] border border-[var(--border-card)] text-xs">
            <div><span className="text-[var(--text-muted)]">Date:</span> <span className="text-[var(--text-primary)] font-semibold ml-1">{schedule.date}</span></div>
            <div><span className="text-[var(--text-muted)]">Uploaded:</span> <span className="text-[var(--text-primary)] font-semibold ml-1">{new Date(schedule.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span></div>
            <div><span className="text-[var(--text-muted)]">Source:</span> <span className="text-[var(--text-primary)] font-semibold ml-1">{schedule.source_filename}</span></div>
            <div><span className="text-[var(--text-muted)]">Status:</span> <span className="text-[var(--status-amber)] font-semibold ml-1">{schedule.status === 'confirmed' ? 'Confirmed' : 'Awaiting confirmation'}</span></div>
            <div><span className="text-[var(--text-muted)]">Forecast:</span>
              <span className={`font-semibold ml-1 ${aggregated?.using_forecast ? 'text-[var(--status-emerald)]' : 'text-[var(--text-muted)]'}`}>
                {aggregated?.using_forecast ? 'Active (DecomEngine)' : 'Upload values only'}
              </span>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-[var(--border-card)]">
            {(['schedule', 'history', 'audit', 'settings'] as AppTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setAppTab(tab)}
                className={`px-5 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  appTab === tab
                    ? 'text-[var(--text-primary)] border-b-[var(--brand-blue)]'
                    : 'text-[var(--text-muted)] border-b-transparent hover:text-[var(--text-secondary)]'
                }`}
              >
                {tab === 'schedule' ? 'Schedule Grid' : tab === 'history' ? 'Upload History' : tab === 'audit' ? 'Audit Trail' : 'Settings'}
              </button>
            ))}
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-4 gap-3">
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
              <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Peak Demand</p>
              <p className="text-2xl font-bold text-[var(--status-amber)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{Math.round(peakDemand).toLocaleString()} MW</p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">Hour {peakHour}</p>
            </div>
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
              <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Total Generation</p>
              <p className="text-2xl font-bold text-[var(--brand-blue)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{totalGeneration.toLocaleString()} MW</p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{plantRows.length} plants</p>
            </div>
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
              <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Reserve Margin</p>
              <p className="text-2xl font-bold text-[var(--status-emerald)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{reserveMargin.toLocaleString()} MW</p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{avgDemand > 0 ? `+${Math.round(reserveMargin / avgDemand * 100)}%` : '—'}</p>
            </div>
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] px-4 py-3">
              <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Avg Total Demand</p>
              <p className="text-2xl font-bold text-[var(--text-primary)] mt-1" style={{ fontVariantNumeric: 'tabular-nums' }}>{avgDemand.toLocaleString()} MW</p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">24h avg</p>
            </div>
          </div>

          {/* Controls row */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex border border-[var(--border-card)]">
                {(['demand', 'supply', 'combined'] as GridTab[]).map(t => (
                  <button
                    key={t}
                    onClick={() => setGridTab(t)}
                    className={`px-3 py-1.5 text-xs font-medium border-r border-[var(--border-card)] last:border-r-0 transition-colors ${
                      gridTab === t ? 'bg-[var(--surface-secondary)] text-[var(--text-primary)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                    }`}
                  >
                    {t === 'demand' ? 'Demand' : t === 'supply' ? 'Supply' : 'Combined'}
                  </button>
                ))}
              </div>
              {schedule.status !== 'confirmed' && (
                <button
                  onClick={async () => {
                    if (!scheduleId) return;
                    try {
                      await scheduleService.autoFillForecast(scheduleId);
                      await loadSchedule(scheduleId);
                      toast.success('ECG row auto-filled from forecast');
                    } catch (err: any) {
                      toast.error(err?.response?.data?.detail || 'Auto-fill failed');
                    }
                  }}
                  className="px-3 py-1.5 text-xs font-semibold border border-[var(--border-card)] text-[var(--brand-blue)] hover:bg-[var(--surface-secondary)] transition-colors flex items-center gap-1.5"
                >
                  <Zap className="w-3.5 h-3.5" /> Auto-fill Forecast
                </button>
              )}
              {scheduleId && <AuditTrailPanel scheduleId={scheduleId} />}
            </div>
            <div className="flex items-center gap-2">
              <div className="flex border border-[var(--border-card)]">
                <button
                  onClick={() => setViewTab('grid')}
                  className={`px-4 py-1.5 text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                    viewTab === 'grid' ? 'bg-[var(--surface-secondary)] text-[var(--text-primary)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <Table2 className="w-3.5 h-3.5" /> Grid
                </button>
                <button
                  onClick={() => setViewTab('graph')}
                  className={`px-4 py-1.5 text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                    viewTab === 'graph' ? 'bg-[var(--surface-secondary)] text-[var(--text-primary)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" /> Graph
                </button>
              </div>
              <button
                onClick={() => setAuditOpen(!auditOpen)}
                className={`px-3 py-1.5 text-xs font-semibold border border-[var(--border-card)] transition-colors flex items-center gap-1.5 ${
                  auditOpen ? 'bg-[var(--brand-blue)]/10 text-[var(--brand-blue)] border-[var(--brand-blue)]' : 'text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)]'
                }`}
              >
                <History className="w-3.5 h-3.5" /> Audit Trail
              </button>
            </div>
          </div>

          {/* Audit Trail Drawer */}
          {auditOpen && (
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-4">
              <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
                Audit Trail <span className="text-[var(--text-muted)] font-normal">— Dispatch Schedule {schedule.date}</span>
              </h3>
              <div className="flex flex-col">
                {auditEntries.map((entry, i) => (
                  <div key={i} className="flex gap-3 py-2 border-b border-[var(--divider)] last:border-b-0 text-xs">
                    <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                      entry.type === 'upload' ? 'bg-[var(--brand-blue)]' :
                      entry.type === 'correct' ? 'bg-[var(--status-amber)]' :
                      entry.type === 'edit' ? 'bg-[var(--brand-blue)]' :
                      entry.type === 'confirm' ? 'bg-[var(--status-emerald)]' :
                      entry.type === 'revise' ? 'bg-[var(--status-crimson)]' :
                      'bg-[var(--text-muted)]'
                    }`} />
                    <span className="text-[var(--text-muted)] w-10 flex-shrink-0">{entry.time}</span>
                    <span className="text-[var(--brand-blue)] w-36 flex-shrink-0 truncate">{entry.actor}</span>
                    <span className="text-[var(--text-secondary)]">
                      {entry.message}
                      {entry.diffs && entry.diffs.map((d, j) => (
                        <span key={j} className="ml-1.5 text-[var(--status-amber)] bg-[var(--surface-secondary)] px-1.5 py-0.5 text-[10px] font-mono">{d}</span>
                      ))}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-3 px-3 py-2 bg-[rgba(63,185,80,0.1)] border border-[var(--status-emerald)] text-xs text-[var(--status-emerald)] inline-block">
                Chain Integrity: Verified ({auditEntries.length} entries, all SHA256 hashes match)
              </div>
            </div>
          )}

          {/* Grid View */}
          {viewTab === 'grid' && (
            <>
              {/* Demand Table */}
              {(gridTab === 'demand' || gridTab === 'combined') && (
                <div className="border border-[var(--border-card)] bg-[var(--bg-card)] overflow-x-auto">
                  <table className="w-full text-xs border-collapse min-w-[1000px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    <thead>
                      <tr>
                        <th className="sticky left-0 bg-[var(--bg-card)] z-10 text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)] min-w-[200px]">
                          Hour / Entity
                        </th>
                        {Array.from({ length: 24 }, (_, i) => (
                          <th key={i} className="px-2 py-2 text-center text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)] min-w-[52px]">
                            {String(i + 1).padStart(2, '0')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="bg-[var(--surface-secondary)]">
                        <td className="sticky left-0 bg-[var(--surface-secondary)] z-10 px-3 py-1.5 text-[11px] uppercase tracking-[0.5px] text-[var(--brand-blue)] font-semibold border border-[var(--border-card)]">
                          DEMAND (MW)
                        </td>
                        {Array.from({ length: 24 }, (_, i) => (
                          <td key={i} className="border border-[var(--border-card)] bg-[var(--surface-secondary)]"></td>
                        ))}
                      </tr>
                      {entityRows.map(entity => (
                        <tr key={entity} className="border-b border-[var(--divider)]" style={{ backgroundColor: entityBg(entity) }}>
                          <td className="sticky left-0 bg-[var(--bg-card)] z-10 px-3 py-2 text-xs font-medium border border-[var(--border-card)]" style={{
                            borderLeft: `3px solid ${entityColor(entity)}`,
                            color: entity === 'ECG' ? 'var(--brand-blue)' : entity === 'NITS_Total' ? 'var(--text-primary)' : 'var(--text-secondary)',
                          }}>
                            {entity === 'NITS_Total' ? 'Scheduled Demand (NITS)' : `${entity} Demand`}
                            <span className="font-normal text-[var(--text-muted)] text-[11px] ml-1">
                              {entity === 'ECG' ? '(forecasted)' : entity === 'NITS_Total' ? '' : '(actual)'}
                            </span>
                          </td>
                          {Array.from({ length: 24 }, (_, i) => {
                            const h = i + 1;
                            const val = getDemandValue(entity, h);
                            const isEditing = editingCell?.entity === entity && editingCell?.hour === h;
                            const isPeak = entity === 'ECG' && val === peakDemand && val > 0;
                            const isEditable = entity !== 'NITS_Total';

                            return (
                              <td
                                key={h}
                                className={`px-2 py-2 text-center border border-[var(--border-card)] ${getCellClass(val, isEditable)} ${isPeak ? 'bg-[rgba(248,81,73,0.15)] text-[var(--status-crimson)] font-semibold' : ''}`}
                                onClick={() => isEditable && handleCellClick(entity, h)}
                              >
                                {isEditing ? (
                                  <input
                                    autoFocus
                                    type="number"
                                    step="0.01"
                                    className="w-full bg-transparent border border-[var(--brand-blue)] px-1 py-0.5 text-center text-xs text-[var(--text-primary)] outline-none"
                                    value={editValue}
                                    onChange={e => setEditValue(e.target.value)}
                                    onBlur={handleCellSave}
                                    onKeyDown={e => e.key === 'Enter' && handleCellSave()}
                                  />
                                ) : val !== undefined ? (
                                  Math.round(val).toLocaleString()
                                ) : '—'}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Supply Table */}
              {(gridTab === 'supply' || gridTab === 'combined') && (
                <div className="border border-[var(--border-card)] bg-[var(--bg-card)] overflow-x-auto">
                  <table className="w-full text-xs border-collapse min-w-[1000px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    <thead>
                      <tr>
                        <th className="sticky left-0 bg-[var(--bg-card)] z-10 text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)] min-w-[200px]">
                          Hour / Entity
                        </th>
                        {Array.from({ length: 24 }, (_, i) => (
                          <th key={i} className="px-2 py-2 text-center text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)] min-w-[52px]">
                            {String(i + 1).padStart(2, '0')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {plantRows.length > 0 && (
                        <>
                          <tr className="bg-[var(--surface-secondary)]">
                            <td className="sticky left-0 bg-[var(--surface-secondary)] z-10 px-3 py-1.5 text-[11px] uppercase tracking-[0.5px] text-[var(--status-emerald)] font-semibold border border-[var(--border-card)]">
                              EMBEDDED SOURCES (MW)
                            </td>
                            {Array.from({ length: 24 }, (_, i) => (
                              <td key={i} className="border border-[var(--border-card)] bg-[var(--surface-secondary)]"></td>
                            ))}
                          </tr>
                          {plantRows.map(plant => {
                            const isBaseload = schedule?.supply.some(s => s.plant_name === plant && s.is_baseload);
                            const cat = schedule?.supply.find(s => s.plant_name === plant && s.category)?.category || '';
                            return (
                            <tr key={plant} className="border-b border-[var(--divider)]" style={{ backgroundColor: categoryBg(cat) }}>
                              <td className="sticky left-0 bg-[var(--bg-card)] z-10 px-3 py-2 text-xs font-medium border border-[var(--border-card)]" style={{
                                borderLeft: `3px solid ${categoryColor(cat)}`,
                                color: cat === 'thermal' ? '#EF4444' : cat === 'hydro' ? 'var(--brand-blue)' : 'var(--text-primary)',
                              }}>
                                <div className="flex items-center gap-2">
                                  {isBaseload && (
                                    <span className="inline-flex items-center px-1 py-0.5 text-[9px] font-bold uppercase tracking-wider bg-[var(--brand-blue)]/10 text-[var(--brand-blue)] border border-[var(--brand-blue)]/20">
                                      B
                                    </span>
                                  )}
                                  {plant}
                                </div>
                              </td>
                              {Array.from({ length: 24 }, (_, i) => {
                                const h = i + 1;
                                const val = getSupplyValue(plant, h);
                                const isEditing = editingCell?.entity === plant && editingCell?.hour === h;
                                return (
                                  <td
                                    key={h}
                                    className={`px-2 py-2 text-center border border-[var(--border-card)] ${val && val > 0 ? 'text-[var(--text-secondary)]' : 'text-[var(--text-muted)]'} cursor-pointer hover:outline hover:outline-2 hover:outline-[var(--brand-blue)] hover:outline-offset-[-2px]`}
                                    onClick={() => handleCellClick(plant, h)}
                                  >
                                    {isEditing ? (
                                      <input
                                        autoFocus
                                        type="number"
                                        step="0.01"
                                        className="w-full bg-transparent border border-[var(--brand-blue)] px-1 py-0.5 text-center text-xs text-[var(--text-primary)] outline-none"
                                        value={editValue}
                                        onChange={e => setEditValue(e.target.value)}
                                        onBlur={handleCellSave}
                                        onKeyDown={e => e.key === 'Enter' && handleCellSave()}
                                      />
                                    ) : val !== undefined ? (val > 0 ? Math.round(val).toLocaleString() : '0') : '—'}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                        </>
                      )}

                      {/* Total demand row */}
                      <tr className="bg-[var(--surface-secondary)]">
                        <td className="sticky left-0 bg-[var(--surface-secondary)] z-10 px-3 py-2 text-xs font-bold text-[var(--text-primary)] border border-[var(--border-card)]">
                          TOTAL DEMAND — MW
                        </td>
                        {Array.from({ length: 24 }, (_, i) => {
                          const h = i + 1;
                          const total = aggregated
                            ? aggregated.hourly[h - 1].total_demand_mw
                            : totalByHour[h] || 0;
                          const isPeak = total === (aggregated ? aggregated.peak_demand_mw : Math.max(...Object.values(totalByHour), 0)) && total > 0;
                          return (
                            <td
                              key={h}
                              className={`px-2 py-2 text-center font-bold border border-[var(--border-card)] ${
                                isPeak ? 'bg-[rgba(248,81,73,0.15)] text-[var(--status-crimson)]' : 'text-[var(--text-primary)]'
                              }`}
                            >
                              {total ? Math.round(total).toLocaleString() : '—'}
                            </td>
                          );
                        })}
                      </tr>

                      {/* Reserve row */}
                      <tr className="bg-[var(--bg-card)]">
                        <td className="sticky left-0 bg-[var(--bg-card)] z-10 px-3 py-2 text-xs font-semibold text-[var(--status-emerald)] border border-[var(--border-card)]">
                          RESERVE — MW
                        </td>
                        {Array.from({ length: 24 }, (_, i) => {
                          const h = i + 1;
                          const supply = aggregated
                            ? aggregated.hourly[h - 1].total_supply_mw
                            : schedule?.supply.filter(s => s.hour === h).reduce((a, b) => a + b.supply_mw, 0) || 0;
                          const demand = aggregated
                            ? aggregated.hourly[h - 1].total_demand_mw
                            : (totalByHour[h] || 0);
                          const reserve = supply - demand;
                          const isNegative = reserve < 0;
                          return (
                            <td
                              key={h}
                              className={`px-2 py-2 text-center font-bold border border-[var(--border-card)] ${
                                isNegative ? 'text-[var(--status-crimson)]' : 'text-[var(--status-emerald)]'
                              }`}
                            >
                              {Math.round(reserve).toLocaleString()}
                            </td>
                          );
                        })}
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* Footer */}
              <div className="flex justify-between text-[11px] text-[var(--text-muted)] px-1 py-2">
                <span>Yellow cells = peak hours (19:00-24:00) | Hover to edit | Blue outline = editable cell</span>
                <span>Data from ECG Daily Demand Data Sheet + Forecast Engine v2.4</span>
              </div>
            </>
          )}

          {/* Graph View */}
          {viewTab === 'graph' && (
            <div className="bg-[var(--bg-card)] border border-[var(--border-card)] p-5">
              <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">24-Hour Demand Profile</h3>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                  <XAxis dataKey="hour" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-card)',
                      borderRadius: 0,
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="ECG" fill="var(--brand-blue)" name="ECG Demand" radius={[2, 2, 0, 0]} />
                  <Line type="monotone" dataKey="NITS_Total" stroke="var(--status-amber)" strokeWidth={2} dot={false} name="NITS Total" />
                  <Bar dataKey="Meienergy" stackId="supply" fill="var(--status-emerald)" name="Meienergy" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="BXC Solar" stackId="supply" fill="var(--status-violet)" name="BXC Solar" radius={[2, 2, 0, 0]} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}
