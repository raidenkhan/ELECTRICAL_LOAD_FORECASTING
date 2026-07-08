'use client';

import React, { useState, useEffect } from 'react';
import {
  Database, FileSpreadsheet, Clock, Edit3, Loader2,
  CheckCircle2, AlertTriangle, BarChart3,
} from 'lucide-react';
import { toast } from 'sonner';
import { api } from '@/lib/api';

type DataTab = 'archive' | 'baseload' | 'history';

export function DataManagement() {
  const [tab, setTab] = useState<DataTab>('archive');

  const tabs: { id: DataTab; label: string; icon: React.ReactNode }[] = [
    { id: 'archive', label: 'Schedule Archive', icon: <Clock className="w-4 h-4" /> },
    { id: 'baseload', label: 'Baseload Registry', icon: <Database className="w-4 h-4" /> },
    { id: 'history', label: 'Historical Data', icon: <BarChart3 className="w-4 h-4" /> },
  ];

  return (
    <div className="flex flex-col gap-4 max-w-[1440px] mx-auto">
      <div className="flex border border-[var(--border-card)]">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-xs font-semibold flex items-center gap-2 transition-colors ${
              tab === t.id ? 'bg-[var(--surface-secondary)] text-[var(--text-primary)] border-b-2 border-[var(--brand-blue)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {tab === 'archive' && <ScheduleArchive />}
      {tab === 'baseload' && <BaseloadRegistry />}
      {tab === 'history' && <HistoricalData />}
    </div>
  );
}

/* ─── Schedule Archive ──────────────────────────────────────────── */

interface ScheduleSummary {
  id: number;
  date: string;
  status: string;
  source_filename: string;
  operator_notes?: string;
  created_at: string;
}

function ScheduleArchive() {
  const [schedules, setSchedules] = useState<ScheduleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<ScheduleSummary[]>('/schedule')
      .then(r => setSchedules(r.data))
      .catch(() => toast.error('Failed to load schedules'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex justify-center py-12"><Loader2 className="w-5 h-5 animate-spin text-[var(--brand-blue)]" /></div>;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)]">
      <div className="px-4 py-3 border-b border-[var(--border-card)]">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">All Dispatch Schedules</h3>
      </div>
      {schedules.length === 0 ? (
        <div className="px-4 py-8 text-center text-xs text-[var(--text-muted)]">No schedules uploaded yet</div>
      ) : (
        <div className="divide-y divide-[var(--divider)]">
          {schedules.map(s => (
            <div key={s.id} className="px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FileSpreadsheet className="w-4 h-4 text-[var(--brand-blue)]" />
                <div>
                  <p className="text-xs font-medium text-[var(--text-primary)]">{s.source_filename}</p>
                  <p className="text-[10px] text-[var(--text-muted)] font-mono">{s.date} — id={s.id}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 border ${
                  s.status === 'confirmed' ? 'border-[var(--status-emerald)] text-[var(--status-emerald)]' : 'border-[var(--status-amber)] text-[var(--status-amber)]'
                }`}>
                  {s.status}
                </span>
                <span className="text-[10px] text-[var(--text-muted)] font-mono">
                  {new Date(s.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Baseload Registry ─────────────────────────────────────────── */

interface BaseloadPlant {
  id: number;
  plant_name: string;
  unit_name: string | null;
  constant_mw: number;
  category: string;
  is_active: boolean;
}

const CATEGORY_COLORS: Record<string, string> = {
  hydro: 'var(--brand-blue)',
  thermal: '#EF4444',
  interconnection: 'var(--status-violet)',
};

function BaseloadRegistry() {
  const [plants, setPlants] = useState<BaseloadPlant[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<{ id: number; val: string } | null>(null);

  useEffect(() => {
    api.get<BaseloadPlant[]>('/baseload/plants')
      .then(r => setPlants(r.data))
      .catch(() => toast.error('Failed to load baseload plants'))
      .finally(() => setLoading(false));
  }, []);

  const saveEdit = async (id: number) => {
    if (!editing || editing.id !== id) return;
    const val = parseFloat(editing.val);
    if (isNaN(val)) { toast.error('Enter valid number'); return; }
    try {
      await api.patch(`/baseload/plants/${id}`, { constant_mw: val });
      setPlants(prev => prev.map(p => p.id === id ? { ...p, constant_mw: val } : p));
      setEditing(null);
      toast.success('Plant updated');
    } catch { toast.error('Failed to update'); }
  };

  if (loading) return <div className="flex justify-center py-12"><Loader2 className="w-5 h-5 animate-spin text-[var(--brand-blue)]" /></div>;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)] overflow-x-auto">
      <div className="px-4 py-3 border-b border-[var(--border-card)]">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">Baseload Plant Registry ({plants.length} plants)</h3>
      </div>
      <table className="w-full text-xs border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
        <thead>
          <tr>
            <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Plant</th>
            <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Unit</th>
            <th className="text-left px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Category</th>
            <th className="text-right px-3 py-2 text-[11px] uppercase tracking-[0.4px] text-[var(--text-muted)] font-semibold border border-[var(--border-card)]">Constant MW</th>
          </tr>
        </thead>
        <tbody>
          {plants.map(p => (
            <tr key={p.id} className="border-b border-[var(--divider)]">
              <td className="px-3 py-2 font-medium text-[var(--text-primary)] border border-[var(--border-card)]">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: CATEGORY_COLORS[p.category] || 'var(--text-secondary)' }} />
                  {p.plant_name}
                </div>
              </td>
              <td className="px-3 py-2 text-[var(--text-secondary)] border border-[var(--border-card)]">{p.unit_name || '—'}</td>
              <td className="px-3 py-2 border border-[var(--border-card)]">
                <span className="text-[10px] font-semibold uppercase" style={{ color: CATEGORY_COLORS[p.category] || 'var(--text-muted)' }}>{p.category}</span>
              </td>
              <td className="px-3 py-2 text-right border border-[var(--border-card)]">
                {editing?.id === p.id ? (
                  <input
                    autoFocus
                    type="number"
                    step="0.1"
                    className="w-20 bg-transparent border border-[var(--brand-blue)] px-1 py-0.5 text-right text-xs text-[var(--text-primary)] outline-none"
                    value={editing.val}
                    onChange={e => setEditing({ ...editing, val: e.target.value })}
                    onBlur={() => saveEdit(p.id)}
                    onKeyDown={e => e.key === 'Enter' && saveEdit(p.id)}
                  />
                ) : (
                  <button
                    onClick={() => setEditing({ id: p.id, val: String(p.constant_mw) })}
                    className="hover:text-[var(--brand-blue)] transition-colors flex items-center justify-end gap-1 w-full"
                  >
                    <Edit3 className="w-3 h-3 opacity-0 group-hover:opacity-100" />
                    {p.constant_mw} MW
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Historical Data ───────────────────────────────────────────── */

function HistoricalData() {
  const [stats, setStats] = useState<{ total_records: number; date_range: string; avg_demand: number } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/data/historical/stats')
      .then(r => setStats(r.data))
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex justify-center py-12"><Loader2 className="w-5 h-5 animate-spin text-[var(--brand-blue)]" /></div>;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)]">
      <div className="px-4 py-3 border-b border-[var(--border-card)]">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">ECG Historical Demand Data</h3>
      </div>
      {stats ? (
        <div className="p-4 grid grid-cols-3 gap-4">
          <div className="px-3 py-2.5 border border-[var(--border-card)]">
            <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Total Records</p>
            <p className="text-lg font-bold text-[var(--text-primary)] mt-1">{stats.total_records.toLocaleString()}</p>
          </div>
          <div className="px-3 py-2.5 border border-[var(--border-card)]">
            <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Date Range</p>
            <p className="text-lg font-bold text-[var(--text-primary)] mt-1">{stats.date_range}</p>
          </div>
          <div className="px-3 py-2.5 border border-[var(--border-card)]">
            <p className="text-[11px] uppercase tracking-[0.8px] text-[var(--text-muted)] font-semibold">Avg Demand</p>
            <p className="text-lg font-bold text-[var(--brand-blue)] mt-1">{Math.round(stats.avg_demand).toLocaleString()} MW</p>
          </div>
        </div>
      ) : (
        <div className="px-4 py-8 text-center text-xs text-[var(--text-muted)]">
          <AlertTriangle className="w-5 h-5 mx-auto mb-2 text-[var(--status-amber)]" />
          Historical data endpoint not available. Run seed script to populate.
        </div>
      )}
    </div>
  );
}
