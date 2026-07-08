'use client';

import React, { useState, useEffect } from 'react';
import { Shield, ShieldCheck, ChevronDown, ChevronRight, Clock, Upload, CheckCircle2, RotateCcw, Edit3, Download, Loader2 } from 'lucide-react';
import { auditService, AuditLogEntry } from '@/services/scheduleService';

const ACTION_ICONS: Record<string, React.ReactNode> = {
  upload: <Upload className="w-3.5 h-3.5" />,
  confirm: <CheckCircle2 className="w-3.5 h-3.5" />,
  revise: <RotateCcw className="w-3.5 h-3.5" />,
  cell_update: <Edit3 className="w-3.5 h-3.5" />,
  forecast_fill: <Download className="w-3.5 h-3.5" />,
};

const ACTION_COLORS: Record<string, string> = {
  upload: 'text-[var(--brand-blue)]',
  confirm: 'text-[var(--status-emerald)]',
  revise: 'text-[var(--status-amber)]',
  cell_update: 'text-[var(--text-secondary)]',
  forecast_fill: 'text-[var(--status-violet)]',
};

const ACTION_BG: Record<string, string> = {
  upload: 'bg-[var(--brand-blue)]/10',
  confirm: 'bg-[var(--status-emerald)]/10',
  revise: 'bg-[var(--status-amber)]/10',
  cell_update: 'bg-[var(--surface-secondary)]',
  forecast_fill: 'bg-[var(--status-violet)]/10',
};

interface AuditTrailPanelProps {
  scheduleId: number;
}

export function AuditTrailPanel({ scheduleId }: AuditTrailPanelProps) {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [chainResult, setChainResult] = useState<{ valid: boolean; count: number; failures: any[] } | null>(null);

  useEffect(() => {
    if (expanded) fetchLogs();
  }, [expanded, scheduleId]);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const [entries, verify] = await Promise.all([
        auditService.getForSchedule(scheduleId),
        auditService.verifyChain(scheduleId),
      ]);
      setLogs(entries);
      setChainResult(verify);
    } catch (e) {
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  if (!expanded) {
    return (
      <button onClick={() => setExpanded(true)} className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold border border-[var(--border-card)] text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)] transition-colors">
        <Shield className="w-3.5 h-3.5" /> Audit Trail <ChevronRight className="w-3 h-3" />
      </button>
    );
  }

  return (
    <div className="border border-[var(--border-card)] bg-[var(--bg-card)]">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--border-card)]">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-[var(--brand-blue)]" />
          <span className="text-xs font-semibold text-[var(--text-primary)]">Audit Trail</span>
          {chainResult && (
            <span className={`flex items-center gap-1 text-[10px] font-mono ${chainResult.valid ? 'text-[var(--status-emerald)]' : 'text-[var(--status-crimson)]'}`}>
              {chainResult.valid ? <ShieldCheck className="w-3 h-3" /> : '!'} Chain {chainResult.valid ? 'Valid' : 'Broken'} ({chainResult.count} entries)
            </span>
          )}
        </div>
        <button onClick={() => setExpanded(false)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>
      <div className="max-h-[300px] overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8"><Loader2 className="w-4 h-4 animate-spin text-[var(--brand-blue)]" /></div>
        ) : logs.length === 0 ? (
          <div className="px-4 py-6 text-xs text-center text-[var(--text-muted)]">No audit entries yet</div>
        ) : (
          <div className="divide-y divide-[var(--divider)]">
            {logs.map((entry) => (
              <div key={entry.id} className="px-4 py-2.5 flex items-start gap-3">
                <div className={`mt-0.5 p-1 rounded-sm ${ACTION_BG[entry.action] || 'bg-[var(--surface-secondary)]'} ${ACTION_COLORS[entry.action] || 'text-[var(--text-secondary)]'}`}>
                  {ACTION_ICONS[entry.action] || <Clock className="w-3.5 h-3.5" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-[var(--text-primary)] uppercase tracking-wider">{entry.action.replace('_', ' ')}</span>
                    <span className="text-[10px] font-mono text-[var(--text-muted)]">{new Date(entry.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 truncate">{entry.description}</p>
                  <div className="flex gap-3 mt-1">
                    <span className="text-[9px] font-mono text-[var(--text-muted)]">hash: {entry.hash.slice(0, 12)}...</span>
                    <span className="text-[9px] font-mono text-[var(--text-muted)]">prev: {entry.previous_hash.slice(0, 12)}...</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
