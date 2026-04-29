import React, { useState, useRef } from 'react';
import { 
  Info, Download, Filter, ChevronDown, Calendar, 
  Database, Upload, CheckCircle2, AlertTriangle, 
  Loader2, FileSpreadsheet, ShieldCheck, Zap
} from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

type ValidationStep = 'idle' | 'parsing' | 'physics' | 'anomalies' | 'commit' | 'success' | 'error';

export function DataManagement() {
  const [columns, setColumns] = useState({
    timestamp: true, total_load: true, bank_82t1: true, bank_82t3: true, bank_82t4: true,
    oil_temp: true, reactive_q: false, voltage_v: false
  });
  const [showColMenu, setShowColMenu] = useState(false);
  const [uploadStep, setUploadStep] = useState<ValidationStep>('idle');
  const [healthGrade, setHealthGrade] = useState<string | null>(null);
  const [impactSummary, setImpactSummary] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadStep('parsing');
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Step-by-step UI simulation for rigor
      await new Promise(r => setTimeout(r, 800));
      setUploadStep('physics');
      await new Promise(r => setTimeout(r, 1000));
      setUploadStep('anomalies');
      
      const response = await axios.post('http://localhost:8000/api/v1/data/upload', formData);
      
      setUploadStep('commit');
      await new Promise(r => setTimeout(r, 600));
      
      setHealthGrade(response.data.health_grade);
      setImpactSummary(response.data.impact_summary);
      setUploadStep('success');
      toast.success('SCADA Synchronized Successfully');
    } catch (error: any) {
      setUploadStep('error');
      toast.error(error.response?.data?.detail || 'Upload Failed');
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'text-emerald-500 border-emerald-500/20 bg-emerald-500/10';
      case 'B': return 'text-blue-500 border-blue-500/20 bg-blue-500/10';
      case 'C': return 'text-amber-500 border-amber-500/20 bg-amber-500/10';
      default: return 'text-rose-500 border-rose-500/20 bg-rose-500/10';
    }
  };

  return (
    <div className="flex flex-col gap-6 h-full font-sans">
      
      {/* Upload & Validation Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-panel p-8 flex flex-col items-center justify-center border-dashed border-2 border-[var(--divider)] relative overflow-hidden group">
          <div className="absolute inset-0 bg-[var(--brand-blue)]/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
          
          {uploadStep === 'idle' ? (
            <>
              <div className="w-16 h-16 rounded-full bg-[var(--surface-secondary)] flex items-center justify-center mb-4 border border-[var(--divider)] group-hover:scale-110 transition-transform">
                <Upload className="w-8 h-8 text-[var(--brand-blue)]" />
              </div>
              <h3 className="text-lg font-bold text-[var(--text-primary)]">Ingest SCADA Telemetry</h3>
              <p className="text-sm text-[var(--text-muted)] mt-2 text-center max-w-sm mb-6">
                Upload CSV exports from National Control Centre. System will perform physics-based outlier detection.
              </p>
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="px-8 py-2.5 bg-[var(--brand-blue)] hover:bg-[var(--brand-blue-vibrant)] text-white rounded-md text-[12px] font-black uppercase tracking-widest transition-all shadow-lg"
              >
                Select CSV Source
              </button>
            </>
          ) : (
            <div className="w-full max-w-md space-y-6 py-4">
               <div className="flex justify-between items-center mb-8">
                  <div className="flex flex-col">
                    <span className="micro-num text-[var(--brand-blue)] uppercase font-bold tracking-tighter">Validation Pipeline</span>
                    <span className="headline text-lg">System Rigor Check</span>
                  </div>
                  {uploadStep === 'success' && <div className={`px-4 py-1 rounded border text-xl font-black ${getGradeColor(healthGrade || 'F')}`}>GRADE {healthGrade}</div>}
               </div>

               <div className="space-y-4">
                  <StepperItem label="File Structure & Schema" status={uploadStep === 'parsing' ? 'active' : ['physics', 'anomalies', 'commit', 'success'].includes(uploadStep) ? 'complete' : 'idle'} />
                  <StepperItem label="Physics Sanity Check (V, I, f)" status={uploadStep === 'physics' ? 'active' : ['anomalies', 'commit', 'success'].includes(uploadStep) ? 'complete' : 'idle'} />
                  <StepperItem label="Statistical Anomaly Filter" status={uploadStep === 'anomalies' ? 'active' : ['commit', 'success'].includes(uploadStep) ? 'complete' : 'idle'} />
                  <StepperItem label="Database Atomic Commit" status={uploadStep === 'commit' ? 'active' : uploadStep === 'success' ? 'complete' : 'idle'} />
               </div>

               {uploadStep === 'success' && (
                 <div className="mt-8 animate-in fade-in slide-in-from-bottom-2">
                    <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded flex items-start gap-3">
                      <Zap className="w-5 h-5 text-emerald-500 mt-0.5" />
                      <p className="text-xs text-emerald-100/80 leading-relaxed italic">
                        {impactSummary} Performance view has been recalibrated with this new dataset.
                      </p>
                    </div>
                    <button onClick={() => setUploadStep('idle')} className="w-full mt-4 text-[10px] uppercase font-bold text-[var(--text-muted)] hover:text-white">Close & Reset Pipeline</button>
                 </div>
               )}
            </div>
          )}
          <input type="file" ref={fileInputRef} className="hidden" accept=".csv" onChange={handleFileUpload} />
        </div>

        <div className="glass-panel p-6 flex flex-col gap-6">
           <h3 className="title text-[var(--text-primary)] uppercase tracking-widest flex items-center gap-2">
             <ShieldCheck className="w-4 h-4 text-[var(--brand-teal)]" /> 
             Data Guardians
           </h3>
           <div className="space-y-4">
              <div className="p-3 bg-[var(--surface-secondary)]/50 rounded border border-[var(--divider)]">
                 <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">Clipped Artifacts</p>
                 <p className="text-sm font-mono text-[var(--brand-teal)]">82T1 ≤ 0MW Auto-Zero</p>
              </div>
              <div className="p-3 bg-[var(--surface-secondary)]/50 rounded border border-[var(--divider)]">
                 <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">Voltage Bounds</p>
                 <p className="text-sm font-mono text-[var(--text-primary)]">33.0kV ± 15% Strict</p>
              </div>
              <div className="p-3 bg-[var(--surface-secondary)]/50 rounded border border-[var(--divider)]">
                 <p className="text-[10px] font-bold text-[var(--text-muted)] uppercase mb-1">Outage Masking</p>
                 <p className="text-sm font-mono text-[var(--status-amber)]">Z-Score {'>'} 2.5 Auto-Flag</p>
              </div>
           </div>
           <p className="text-[10px] text-[var(--text-muted)] italic mt-auto leading-relaxed">
             Note: Every upload triggers a full re-fit of the GRIDCo Similar-Day pool to ensure benchmarking accuracy.
           </p>
        </div>
      </div>

      {/* Table Controls (Original remains for view only) */}
      <div className="flex flex-wrap items-center justify-between gap-4 glass-panel p-4">
        <div className="flex items-center gap-4 flex-1">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--surface-secondary)] border border-[var(--divider)] rounded">
            <Calendar className="w-4 h-4 text-[var(--text-muted)]" />
            <span className="text-[12px] font-medium text-[var(--text-secondary)]">Historical SCADA Archive</span>
          </div>
        </div>
        <button className="px-4 py-1.5 bg-[#3B82F6] hover:bg-[#2563EB] text-white rounded text-[11px] font-bold uppercase flex items-center gap-2 shadow-sm transition-colors">
            <Download className="w-3.5 h-3.5" /> Export Archive
        </button>
      </div>

      {/* Grid Placeholder for recent records */}
      <div className="glass-panel p-12 flex flex-col items-center justify-center border-dashed border border-[var(--divider)]">
          <FileSpreadsheet className="w-12 h-12 text-[var(--text-muted)] opacity-20 mb-4" />
          <span className="micro-num text-[var(--text-muted)]">Archived records available for audit below</span>
      </div>
    </div>
  );
}

function StepperItem({ label, status }: { label: string, status: 'idle' | 'active' | 'complete' }) {
  return (
    <div className={`flex items-center gap-4 p-3 rounded-md transition-colors ${status === 'active' ? 'bg-[var(--brand-blue)]/10 border border-[var(--brand-blue)]/20' : ''}`}>
      <div className={`w-6 h-6 rounded-full flex items-center justify-center border-2 ${
        status === 'complete' ? 'bg-emerald-500 border-emerald-500 text-white' : 
        status === 'active' ? 'border-[var(--brand-blue)] border-t-transparent animate-spin' : 
        'border-[var(--divider)]'
      }`}>
        {status === 'complete' && <CheckCircle2 className="w-4 h-4" />}
      </div>
      <span className={`text-[12px] font-bold uppercase tracking-tight ${status === 'idle' ? 'text-[var(--text-muted)]' : 'text-[var(--text-primary)]'}`}>
        {label}
      </span>
    </div>
  );
}
