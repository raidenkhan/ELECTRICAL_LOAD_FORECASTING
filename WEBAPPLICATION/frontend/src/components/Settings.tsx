'use client';

import { useState } from 'react';
import {
    User,
    Bell,
    Shield,
    Database,
    Zap,
    Save,
    RefreshCw,
    AlertTriangle,
    CheckCircle,
    Settings as SettingsIcon,
    ChevronRight,
    Search
} from 'lucide-react';

export function Settings() {
    const [activeSection, setActiveSection] = useState('profile');
    const [toast, setToast] = useState<{ message: string, type: 'success' | 'error' } | null>(null);

    const showToast = (message: string, type: 'success' | 'error' = 'success') => {
        setToast({ message, type });
        setTimeout(() => setToast(null), 3000);
    };

    const sections = [
        { id: 'profile', label: 'User Profile', icon: User },
        { id: 'notifications', label: 'Alert Config', icon: Bell },
        { id: 'security', label: 'Security & AAA', icon: Shield },
        { id: 'data', label: 'Model Registry', icon: Database },
        { id: 'system', label: 'Nodes & Sync', icon: SettingsIcon }
    ];

    return (
        <div className="flex flex-col gap-8 h-full font-sans">
            {toast && (
                <div className={`fixed bottom-8 right-8 px-6 py-4 rounded-lg shadow-2xl z-50 flex items-center gap-3 animate-in slide-in-from-bottom-5 border
                    ${toast.type === 'success' ? 'bg-[var(--bg-card)] border-[var(--status-emerald)] text-[var(--status-emerald)]' : 'bg-[var(--bg-card)] border-[var(--status-crimson)] text-[var(--status-crimson)]'}
                `}>
                    {toast.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                    <span className="font-bold text-[12px] uppercase tracking-widest">{toast.message}</span>
                </div>
            )}

            {/* Header */}
            <div className="flex flex-col gap-1">
                <h1 className="display-num text-[var(--text-primary)]">System Configuration</h1>
                <p className="caption text-[var(--text-muted)] uppercase tracking-[0.3em]">
                   Operational Parameters / User Rights / Registry Updates
                </p>
            </div>

            <div className="flex gap-8">
                {/* Sidebar Navigation */}
                <div className="w-[220px] shrink-0">
                    <div className="flex flex-col gap-1">
                        {sections.map((section) => {
                            const Icon = section.icon;
                            const isActive = activeSection === section.id;
                            return (
                                <button
                                    key={section.id}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`flex items-center gap-3 px-4 py-3 rounded text-left transition-all
                                        ${isActive 
                                          ? 'bg-[var(--brand-indigo)]/10 text-[var(--brand-indigo)] border-l-2 border-[var(--brand-indigo)]' 
                                          : 'text-[var(--text-secondary)] hover:bg-[var(--surface-secondary)]'}
                                    `}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="text-[13px] font-bold uppercase tracking-tight">{section.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 max-w-[900px]">
                    <div className="glass-panel p-8">
                       {activeSection === 'profile' && <ProfileSettings showToast={showToast} />}
                       {activeSection === 'notifications' && <NotificationSettings showToast={showToast} />}
                       {activeSection === 'data' && <DataSettings showToast={showToast} />}
                       {activeSection === 'security' && <SecuritySettings showToast={showToast} />}
                    </div>
                </div>
            </div>
        </div>
    );
}

function ProfileSettings({ showToast }: any) {
    return (
        <div className="flex flex-col gap-8">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold border-b border-[var(--divider)] pb-4">User Identity</h2>
            <div className="grid grid-cols-2 gap-6">
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">Full Name</label>
                    <input className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none focus:border-[var(--brand-indigo)]" defaultValue="John Mensah" />
                </div>
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">Email</label>
                    <input className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none" defaultValue="john.mensah@gridco.gov.gh" />
                </div>
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">Organization</label>
                    <input className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none" defaultValue="Ghana Grid Company (GRIDCo)" disabled />
                </div>
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">Role</label>
                    <div className="px-3 py-2 bg-[var(--divider)] rounded text-[12px] font-bold text-[var(--text-secondary)] w-fit mt-1">SENIOR OPERATOR (GAR)</div>
                </div>
            </div>
            <button onClick={() => showToast('Profile Securely Updated')} className="mt-4 px-6 py-2 bg-[var(--brand-indigo)] text-white rounded text-[12px] font-bold uppercase tracking-widest w-fit hover:bg-[var(--brand-indigo)]/90 transition-all flex items-center gap-2">
                <Save className="w-4 h-4" /> Save Metadata
            </button>
        </div>
    );
}

function NotificationSettings({ showToast }: any) {
   return (
      <div className="flex flex-col gap-8">
         <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold border-b border-[var(--divider)] pb-4">Critical Alert Logic</h2>
         <div className="space-y-6">
            {[
               { label: 'Exceedance Warnings', desc: 'Alert when load exceeds 1,600 MW threshold', enabled: true },
               { label: 'Decomp Drift', desc: 'Notify on significant residual scaling variance (>15%)', enabled: true },
               { label: 'SCADA Timeout', desc: 'Alert on interval sync loss (>5min)', enabled: true },
               { label: 'Unit Trips', desc: 'Immediate notification on generation unit offline state', enabled: false }
            ].map(item => (
               <div key={item.label} className="flex items-center justify-between p-4 bg-[var(--surface-secondary)] rounded-lg border border-[var(--divider)]">
                  <div className="flex flex-col">
                     <span className="text-[13px] font-bold text-[var(--text-primary)] uppercase tracking-tight">{item.label}</span>
                     <span className="text-[11px] text-[var(--text-muted)]">{item.desc}</span>
                  </div>
                  <div 
                     onClick={() => showToast(`Status: ${item.label} Updated`)}
                     className={`w-10 h-5 rounded-full relative cursor-pointer transition-all ${item.enabled ? 'bg-[var(--brand-indigo)]' : 'bg-[var(--divider)]'}`}>
                     <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${item.enabled ? 'left-6' : 'left-1'}`} />
                  </div>
               </div>
            ))}
         </div>
      </div>
   );
}

function DataSettings({ showToast }: any) {
   return (
       <div className="flex flex-col gap-8">
         <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold border-b border-[var(--divider)] pb-4">Model Engine Registry</h2>
         <div className="space-y-6">
            <div className="flex flex-col gap-4">
               <span className="caption text-[var(--text-muted)] uppercase font-bold">Active Forecast Kernel</span>
               <select className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none appearance-none">
                  <option>Multiplicative Decomp LSTM v3.2 (Champion)</option>
                  <option>SARIMA-Hybrid v1.2</option>
                  <option>Ridge Regression Linear (Baseline)</option>
               </select>
            </div>
            
            <div className="p-4 bg-[var(--status-crimson)]/5 border border-[var(--status-crimson)]/10 rounded">
               <h4 className="text-[12px] font-bold text-[var(--status-crimson)] uppercase mb-2">Danger Zone: Registry Reset</h4>
               <p className="text-[11px] text-[var(--text-secondary)] mb-4">Restore system to factory Community Baseline (80MW). All training weights will be purged.</p>
               <button onClick={() => confirm('Purge Registry?') && showToast('System Reset', 'error')} className="px-4 py-2 border border-[var(--status-crimson)] text-[var(--status-crimson)] rounded text-[11px] font-bold uppercase">
                  Revert to Baseline
               </button>
            </div>
         </div>
      </div>
   );
}

function SecuritySettings({ showToast }: any) {
    return (
        <div className="flex flex-col gap-8">
            <h2 className="title text-[var(--text-primary)] uppercase tracking-widest font-bold border-b border-[var(--divider)] pb-4">Security & Access Control</h2>
            <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">Current Pass</label>
                    <input className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none" type="password" placeholder="••••••••" />
                </div>
                <div className="flex flex-col gap-1.5">
                    <label className="caption text-[var(--text-muted)] uppercase font-bold">New Policy Pass</label>
                    <input className="bg-[var(--surface-secondary)] border border-[var(--divider)] rounded p-2 text-[14px] text-[var(--text-primary)] outline-none" type="password" placeholder="••••••••" />
                </div>
                <div className="p-4 bg-[var(--brand-indigo)]/5 border border-[var(--divider)] rounded flex items-center justify-between">
                   <div className="flex items-center gap-3">
                      <Shield className="w-5 h-5 text-[var(--brand-indigo)]" />
                      <div className="flex flex-col">
                         <span className="text-[12px] font-bold text-[var(--text-primary)]">MFA Multi-Factor Auth</span>
                         <span className="text-[10px] text-[var(--text-muted)] uppercase">Status: Not registered</span>
                      </div>
                   </div>
                   <button onClick={() => showToast('MFA Activation Syncing')} className="px-4 py-1.5 bg-[var(--brand-indigo)] text-white rounded text-[10px] font-bold uppercase">Enable 2FA</button>
                </div>
            </div>
        </div>
    );
}
