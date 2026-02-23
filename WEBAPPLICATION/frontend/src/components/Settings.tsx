'use client';

import { useState } from 'react';
import {
    User,
    Bell,
    Shield,
    Database,
    Zap,
    Mail,
    Globe,
    Clock,
    Save,
    RefreshCw,
    AlertTriangle,
    CheckCircle,
    Settings as SettingsIcon
} from 'lucide-react';

export function Settings() {
    const [activeSection, setActiveSection] = useState('profile');
    const [notificationsEnabled, setNotificationsEnabled] = useState(true);
    const [emailAlerts, setEmailAlerts] = useState(true);
    const [criticalOnly, setCriticalOnly] = useState(false);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [refreshInterval, setRefreshInterval] = useState('15');
    const [toast, setToast] = useState<{ message: string, type: 'success' | 'error' } | null>(null);

    const showToast = (message: string, type: 'success' | 'error' = 'success') => {
        setToast({ message, type });
        setTimeout(() => setToast(null), 3000);
    };

    const sections = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'security', label: 'Security', icon: Shield },
        { id: 'data', label: 'Data & Models', icon: Database },
        { id: 'system', label: 'System', icon: SettingsIcon }
    ];

    return (
        <div className="space-y-6 relative">
            {toast && (
                <div className={`fixed bottom-6 right-6 px-6 py-4 rounded-lg shadow-lg z-50 flex items-center gap-3 transition-all duration-300 animate-in slide-in-from-bottom-5 fade-in ${toast.type === 'success' ? 'bg-[#DCFCE7] text-[#166534] border border-[#166534]/20' : 'bg-[#FEE2E2] text-[#991B1B] border border-[#991B1B]/20'
                    }`}>
                    {toast.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                    <span className="font-semibold text-sm">{toast.message}</span>
                </div>
            )}
            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
                    Settings
                </h2>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    Manage your account preferences and system configuration
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar Navigation */}
                <div
                    className="lg:col-span-1 glass-morphism"
                    style={{ height: 'fit-content' }}
                >

                    <div className="p-4">
                        <h3 className="text-xs font-semibold mb-3 px-3" style={{ color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
                            SETTINGS
                        </h3>
                        <nav className="space-y-1">
                            {sections.map((section) => {
                                const Icon = section.icon;
                                const isActive = activeSection === section.id;

                                return (
                                    <button
                                        key={section.id}
                                        onClick={() => setActiveSection(section.id)}
                                        className="w-full flex items-center gap-3 px-3 py-2.5 text-left transition-all duration-200"
                                        style={{
                                            backgroundColor: isActive ? 'var(--hover-bg)' : 'transparent',
                                            color: isActive ? 'var(--lime-primary)' : 'var(--text-secondary)',
                                            fontWeight: isActive ? 600 : 500,
                                            fontSize: '14px',
                                            borderLeft: isActive ? '2px solid var(--lime-primary)' : '2px solid transparent'
                                        }}
                                        onMouseEnter={(e) => {
                                            if (!isActive) {
                                                e.currentTarget.style.backgroundColor = 'var(--hover-bg)';
                                            }
                                        }}

                                        onMouseLeave={(e) => {
                                            if (!isActive) {
                                                e.currentTarget.style.backgroundColor = 'transparent';
                                            }
                                        }}
                                    >
                                        <Icon className="w-5 h-5" />
                                        {section.label}
                                    </button>
                                );
                            })}
                        </nav>
                    </div>
                </div>

                {/* Content Area */}
                <div className="lg:col-span-3">
                    {activeSection === 'profile' && <ProfileSettings showToast={showToast} />}
                    {activeSection === 'notifications' && (
                        <NotificationSettings
                            notificationsEnabled={notificationsEnabled}
                            setNotificationsEnabled={setNotificationsEnabled}
                            emailAlerts={emailAlerts}
                            setEmailAlerts={setEmailAlerts}
                            criticalOnly={criticalOnly}
                            setCriticalOnly={setCriticalOnly}
                            showToast={showToast}
                        />
                    )}
                    {activeSection === 'security' && <SecuritySettings showToast={showToast} />}
                    {activeSection === 'data' && <DataSettings showToast={showToast} />}
                    {activeSection === 'system' && (
                        <SystemSettings
                            autoRefresh={autoRefresh}
                            setAutoRefresh={setAutoRefresh}
                            refreshInterval={refreshInterval}
                            setRefreshInterval={setRefreshInterval}
                            showToast={showToast}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}

interface SettingsProps {
    showToast: (msg: string, type: 'success' | 'error') => void;
}

function ProfileSettings({ showToast }: SettingsProps) {
    const [isLoading, setIsLoading] = useState(false);
    const [formData, setFormData] = useState({
        name: 'John Mensah',
        email: 'john.mensah@gridco.gov.gh',
        org: 'Ghana Grid Company (GRIDCo)',
        role: 'Senior Operator',
        region: 'Greater Accra Region (GAR)'
    });

    const handleSave = () => {
        setIsLoading(true);
        // Simulate API call
        setTimeout(() => {
            setIsLoading(false);
            showToast('Profile updated successfully', 'success');
        }, 1000);
    };

    return (
        <div className="space-y-6">
            <SettingsCard title="Personal Information" description="Update your profile details">
                <div className="space-y-4">
                    <FormField label="Full Name" value={formData.name} onChange={(v: string) => setFormData({ ...formData, name: v })} />
                    <FormField label="Email Address" value={formData.email} onChange={(v: string) => setFormData({ ...formData, email: v })} />
                    <FormField label="Organization" value={formData.org} onChange={(v: string) => setFormData({ ...formData, org: v })} />
                    <FormField label="Role" value={formData.role} onChange={(v: string) => setFormData({ ...formData, role: v })} />
                    <FormField label="Region" value={formData.region} onChange={(v: string) => setFormData({ ...formData, region: v })} />
                </div>
                <div className="flex gap-3 mt-6">
                    <button
                        onClick={handleSave}
                        disabled={isLoading}
                        className="px-6 py-2 text-sm font-bold tracking-widest uppercase border transition-all duration-300 active:scale-95 flex items-center gap-2 disabled:opacity-50"
                        style={{
                            borderColor: 'var(--lime-primary)',
                            color: 'var(--lime-primary)',
                            fontFamily: 'var(--font-geist-mono)'
                        }}

                        onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#1e40af';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#2563EB';
                        }}
                    >
                        {isLoading ? <RefreshCw className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 inline mr-2" />}
                        {isLoading ? 'Saving...' : 'Save Changes'}
                    </button>
                    <button
                        className="px-4 py-2 text-sm font-bold tracking-widest uppercase border transition-all duration-300 active:scale-95"
                        style={{
                            color: 'var(--text-tertiary)',
                            borderColor: 'var(--border-primary)',
                            fontFamily: 'var(--font-geist-mono)'
                        }}
                        onClick={() => showToast('Changes reverted', 'error')}
                    >
                        Cancel
                    </button>
                </div>
            </SettingsCard>
        </div>
    );
}

function NotificationSettings({
    notificationsEnabled,
    setNotificationsEnabled,
    emailAlerts,
    setEmailAlerts,
    criticalOnly,
    setCriticalOnly
}: any) {
    return (
        <div className="space-y-6">
            <SettingsCard title="Alert Preferences" description="Configure how you receive notifications">
                <div className="space-y-4">
                    <ToggleField
                        label="Enable Notifications"
                        description="Receive alerts for forecast anomalies and system events"
                        checked={notificationsEnabled}
                        onChange={setNotificationsEnabled}
                    />
                    <ToggleField
                        label="Email Alerts"
                        description="Send notifications to your email address"
                        checked={emailAlerts}
                        onChange={setEmailAlerts}
                    />
                    <ToggleField
                        label="Critical Alerts Only"
                        description="Only receive notifications for critical events"
                        checked={criticalOnly}
                        onChange={setCriticalOnly}
                    />
                </div>
            </SettingsCard>

            <SettingsCard title="Alert Types" description="Choose which events trigger notifications">
                <div className="space-y-3">
                    <CheckboxField label="Peak forecast exceeds capacity" checked={true} />
                    <CheckboxField label="Model performance degradation" checked={true} />
                    <CheckboxField label="Data quality issues" checked={true} />
                    <CheckboxField label="System health warnings" checked={false} />
                    <CheckboxField label="Model retraining completed" checked={false} />
                </div>
            </SettingsCard>
        </div>
    );
}

function SecuritySettings({ showToast }: SettingsProps) {
    const [passwords, setPasswords] = useState({ current: '', new: '', confirm: '' });
    const [is2FAEnabled, setIs2FAEnabled] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    const handleUpdatePassword = () => {
        if (!passwords.current || !passwords.new || !passwords.confirm) {
            showToast('Please fill all password fields', 'error');
            return;
        }
        if (passwords.new !== passwords.confirm) {
            showToast('New passwords do not match', 'error');
            return;
        }
        setIsLoading(true);
        setTimeout(() => {
            setIsLoading(false);
            setPasswords({ current: '', new: '', confirm: '' });
            showToast('Password updated successfully', 'success');
        }, 1500);
    };

    return (
        <div className="space-y-6">
            <SettingsCard title="Password" description="Update your password">
                <div className="space-y-4">
                    <FormField
                        label="Current Password"
                        type="password"
                        placeholder="Enter current password"
                        value={passwords.current}
                        onChange={(v: string) => setPasswords({ ...passwords, current: v })}
                    />
                    <FormField
                        label="New Password"
                        type="password"
                        placeholder="Enter new password"
                        value={passwords.new}
                        onChange={(v: string) => setPasswords({ ...passwords, new: v })}
                    />
                    <FormField
                        label="Confirm New Password"
                        type="password"
                        placeholder="Confirm new password"
                        value={passwords.confirm}
                        onChange={(v: string) => setPasswords({ ...passwords, confirm: v })}
                    />
                </div>
                <button
                    onClick={handleUpdatePassword}
                    disabled={isLoading}
                    className="px-6 py-2 text-sm font-bold tracking-widest uppercase border mt-6 transition-all duration-300 active:scale-95 disabled:opacity-50 flex items-center gap-2"
                    style={{
                        borderColor: 'var(--lime-primary)',
                        color: 'var(--lime-primary)',
                        fontFamily: 'var(--font-geist-mono)'
                    }}
                >
                    {isLoading && <RefreshCw className="w-4 h-4 animate-spin" />}
                    {isLoading ? 'Updating...' : 'Update Password'}
                </button>
            </SettingsCard>

            <SettingsCard title="Two-Factor Authentication" description="Add an extra layer of security">
                <div className="flex items-start gap-4 p-4 border-l-4 transition-colors" style={{
                    backgroundColor: is2FAEnabled ? 'rgba(220, 252, 231, 0.5)' : '#FEF3C7',
                    borderLeftColor: is2FAEnabled ? '#059669' : '#F59E0B'
                }}>
                    {is2FAEnabled ? (
                        <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: '#059669' }} />
                    ) : (
                        <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: '#F59E0B' }} />
                    )}
                    <div>
                        <p className="text-sm font-semibold mb-1" style={{ color: is2FAEnabled ? '#065F46' : '#92400E' }}>
                            {is2FAEnabled ? 'Two-Factor Authentication Enabled' : 'Two-Factor Authentication Not Enabled'}
                        </p>
                        <p className="text-sm mb-3" style={{ color: is2FAEnabled ? '#065F46' : '#92400E' }}>
                            {is2FAEnabled ? 'Your account is protected with an additional security layer.' : 'Protect your account with an additional security layer.'}
                        </p>
                        <button
                            onClick={() => {
                                setIs2FAEnabled(!is2FAEnabled);
                                showToast(is2FAEnabled ? '2FA disabled' : '2FA enabled', is2FAEnabled ? 'error' : 'success');
                            }}
                            className="px-4 py-2 text-sm font-semibold transition-colors rounded"
                            style={{
                                backgroundColor: is2FAEnabled ? '#DC2626' : '#F59E0B',
                                color: 'white'
                            }}
                        >
                            {is2FAEnabled ? 'Disable 2FA' : 'Enable 2FA'}
                        </button>
                    </div>
                </div>
            </SettingsCard>

            <SettingsCard title="Active Sessions" description="Manage your active login sessions">
                <div className="space-y-3">
                    <SessionItem
                        device="Chrome on Windows"
                        location="Accra, Ghana"
                        lastActive="Active now"
                        isCurrent={true}
                    />
                    <SessionItem
                        device="Safari on iPhone"
                        location="Accra, Ghana"
                        lastActive="2 hours ago"
                        isCurrent={false}
                        onRevoke={() => showToast('Session revoked', 'success')}
                    />
                </div>
            </SettingsCard>
        </div>
    );
}

function DataSettings({ showToast }: SettingsProps) {
    const [modelConfig, setModelConfig] = useState({
        activeModel: 'LightGBM Ensemble v3.2',
        horizon: '24 hours',
        frequency: '15 minutes'
    });
    const [retention, setRetention] = useState({
        historical: '2 years',
        archive: '90 days'
    });

    const handleModelChange = (key: string, value: string) => {
        setModelConfig(prev => ({ ...prev, [key]: value }));
        showToast(`Configuration updated: ${value}`, 'success');
    };

    const handleRetentionChange = (key: string, value: string) => {
        setRetention(prev => ({ ...prev, [key]: value }));
        showToast(`Retention policy updated: ${value}`, 'success');
    };

    return (
        <div className="space-y-6">
            <SettingsCard title="Model Configuration" description="Configure forecasting models">
                <div className="space-y-4">
                    <SelectField
                        label="Active Model"
                        value={modelConfig.activeModel}
                        options={['LightGBM Ensemble v3.2', 'LSTM v2.1', 'Transformer v1.5']}
                        onChange={(v: string) => handleModelChange('activeModel', v)}
                    />
                    <SelectField
                        label="Forecast Horizon"
                        value={modelConfig.horizon}
                        options={['6 hours', '12 hours', '24 hours', '48 hours']}
                        onChange={(v: string) => handleModelChange('horizon', v)}
                    />
                    <SelectField
                        label="Update Frequency"
                        value={modelConfig.frequency}
                        options={['5 minutes', '15 minutes', '30 minutes', '1 hour']}
                        onChange={(v: string) => handleModelChange('frequency', v)}
                    />
                </div>
            </SettingsCard>

            <SettingsCard title="Data Sources" description="Manage connected data feeds">
                <div className="space-y-3">
                    <DataSourceItem
                        name="Internal Database"
                        status="connected"
                        lastSync="2 minutes ago"
                    />
                    <DataSourceItem
                        name="Weather API"
                        status="connected"
                        lastSync="5 minutes ago"
                    />
                    <DataSourceItem
                        name="Maintenance System"
                        status="disconnected"
                        lastSync="Never"
                    />
                </div>
            </SettingsCard>

            <SettingsCard title="Data Retention" description="Configure data storage policies">
                <div className="space-y-4">
                    <SelectField
                        label="Historical Data Retention"
                        value={retention.historical}
                        options={['6 months', '1 year', '2 years', '5 years', 'Indefinite']}
                        onChange={(v: string) => handleRetentionChange('historical', v)}
                    />
                    <SelectField
                        label="Forecast Archive"
                        value={retention.archive}
                        options={['30 days', '90 days', '180 days', '1 year']}
                        onChange={(v: string) => handleRetentionChange('archive', v)}
                    />
                </div>
            </SettingsCard>
        </div>
    );
}

interface SystemSettingsProps {
    autoRefresh: boolean;
    setAutoRefresh: (v: boolean) => void;
    refreshInterval: string;
    setRefreshInterval: (v: string) => void;
    showToast: (msg: string, type: 'success' | 'error') => void;
}

function SystemSettings({ autoRefresh, setAutoRefresh, refreshInterval, setRefreshInterval, showToast }: SystemSettingsProps) {
    const [animationsEnabled, setAnimationsEnabled] = useState(true);
    const [preloadData, setPreloadData] = useState(true);
    const [timeZone, setTimeZone] = useState('GMT (UTC+0)');
    const [dateFormat, setDateFormat] = useState('DD/MM/YYYY');

    return (
        <div className="space-y-6">
            <SettingsCard title="Display Preferences" description="Customize dashboard appearance">
                <div className="space-y-4">
                    <ToggleField
                        label="Auto-Refresh Dashboard"
                        description="Automatically update data at specified intervals"
                        checked={autoRefresh}
                        onChange={(v: boolean) => {
                            setAutoRefresh(v);
                            showToast(v ? 'Auto-refresh enabled' : 'Auto-refresh disabled', v ? 'success' : 'error');
                        }}
                    />
                    {autoRefresh && (
                        <SelectField
                            label="Refresh Interval"
                            value={refreshInterval}
                            options={['15', '30', '60', '120']}
                            onChange={(v: string) => {
                                setRefreshInterval(v);
                                showToast(`Refresh interval set to ${v}s`, 'success');
                            }}
                            unit="seconds"
                        />
                    )}
                    <SelectField
                        label="Time Zone"
                        value={timeZone}
                        options={['GMT (UTC+0)', 'WAT (UTC+1)', 'EAT (UTC+3)']}
                        onChange={(v: string) => {
                            setTimeZone(v);
                            showToast(`Time zone updated to ${v}`, 'success');
                        }}
                    />
                    <SelectField
                        label="Date Format"
                        value={dateFormat}
                        options={['DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY-MM-DD']}
                        onChange={(v: string) => {
                            setDateFormat(v);
                            showToast(`Date format updated to ${v}`, 'success');
                        }}
                    />
                </div>
            </SettingsCard>

            <SettingsCard title="Performance" description="Optimize system performance">
                <div className="space-y-4">
                    <ToggleField
                        label="Enable Chart Animations"
                        description="Show smooth transitions in charts and graphs"
                        checked={animationsEnabled}
                        onChange={(v: boolean) => {
                            setAnimationsEnabled(v);
                            showToast(v ? 'Animations enabled' : 'Animations disabled', 'success');
                        }}
                    />
                    <ToggleField
                        label="Preload Historical Data"
                        description="Load historical data in background for faster access"
                        checked={preloadData}
                        onChange={(v: boolean) => {
                            setPreloadData(v);
                            showToast(v ? 'Preloading enabled' : 'Preloading disabled', 'success');
                        }}
                    />
                </div>
            </SettingsCard>

            <SettingsCard title="System Information" description="View system details">
                <div className="grid grid-cols-2 gap-4">
                    <InfoItem label="Version" value="3.2.1" />
                    <InfoItem label="Environment" value="Production" />
                    <InfoItem label="API Status" value="Operational" status="success" />
                    <InfoItem label="Last Update" value="2024-02-04" />
                </div>
            </SettingsCard>
        </div>
    );
}

// Helper Components
function SettingsCard({ title, description, children }: any) {
    return (
        <div
            className="glass-morphism"
        >
            <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--border-primary)' }}>
                <h3 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {title}
                </h3>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    {description}
                </p>
            </div>
            <div className="p-6">
                {children}
            </div>
        </div>
    );
}


function FormField({ label, value, type = 'text', placeholder, onChange }: any) {
    return (
        <div>
            <label className="block text-sm font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
                {label}
            </label>
            <input
                type={type}
                value={value}
                onChange={(e) => onChange?.(e.target.value)}
                placeholder={placeholder}
                className="w-full px-4 py-2.5 transition-all duration-200 border"
                style={{
                    backgroundColor: 'var(--input-bg)',
                    borderColor: 'var(--input-border)',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                }}
            />
        </div>
    );
}


function ToggleField({ label, description, checked, onChange }: any) {
    return (
        <div className="flex items-start justify-between py-3 border-b last:border-b-0" style={{ borderColor: 'var(--border-primary)' }}>
            <div className="flex-1">
                <p className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                    {label}
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {description}
                </p>
            </div>
            <button
                onClick={() => onChange(!checked)}
                className="relative w-11 h-6 transition-colors flex-shrink-0"
                style={{
                    backgroundColor: checked ? 'var(--lime-primary)' : 'var(--input-border)',
                    borderRadius: '12px'
                }}
            >
                <div
                    className="absolute top-1 w-4 h-4 bg-white rounded-full transition-transform"
                    style={{
                        left: checked ? '24px' : '4px'
                    }}
                />
            </button>
        </div>
    );
}


function CheckboxField({ label, checked }: any) {
    return (
        <label className="flex items-center gap-3 py-2 cursor-pointer">
            <input
                type="checkbox"
                defaultChecked={checked}
                className="w-4 h-4 cursor-pointer"
                style={{ accentColor: '#2563EB' }}
            />
            <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                {label}
            </span>
        </label>
    );
}

function SelectField({ label, value, options, onChange, unit }: any) {
    return (
        <div>
            <label className="block text-sm font-semibold mb-2" style={{ color: 'var(--text-secondary)' }}>
                {label}
            </label>
            <select
                value={value}
                onChange={(e) => onChange?.(e.target.value)}
                className="w-full px-4 py-2.5 transition-all duration-200 border"
                style={{
                    backgroundColor: 'var(--input-bg)',
                    borderColor: 'var(--input-border)',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                }}
            >
                {options.map((option: string) => (
                    <option key={option} value={option}>
                        {option}{unit ? ` ${unit}` : ''}
                    </option>
                ))}
            </select>
        </div>
    );
}


function SessionItem({ device, location, lastActive, isCurrent, onRevoke }: any) {
    return (
        <div className="flex items-start justify-between py-3 border-b last:border-b-0" style={{ borderColor: '#F1F5F9' }}>
            <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                    <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {device}
                    </p>
                    {isCurrent && (
                        <span className="px-2 py-0.5 text-xs font-semibold" style={{
                            backgroundColor: '#DCFCE7',
                            color: '#059669'
                        }}>
                            Current
                        </span>
                    )}
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {location} • {lastActive}
                </p>
            </div>
            {!isCurrent && (
                <button
                    onClick={onRevoke}
                    className="text-xs font-semibold px-3 py-1.5 border transition-colors"
                    style={{
                        color: '#DC2626',
                        borderColor: '#DC2626'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#FEE2E2';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                >
                    Revoke
                </button>
            )}
        </div>
    );
}

function DataSourceItem({ name, status, lastSync }: any) {
    const statusConfig = {
        connected: { color: '#059669', bg: '#DCFCE7', label: 'Connected' },
        disconnected: { color: '#DC2626', bg: '#FEE2E2', label: 'Disconnected' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];

    return (
        <div className="flex items-center justify-between py-3 border-b last:border-b-0" style={{ borderColor: '#F1F5F9' }}>
            <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: config.color }} />
                <div>
                    <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {name}
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Last sync: {lastSync}
                    </p>
                </div>
            </div>
            <span className="px-2 py-1 text-xs font-semibold" style={{
                backgroundColor: config.bg,
                color: config.color
            }}>
                {config.label}
            </span>
        </div>
    );
}

function InfoItem({ label, value, status }: any) {
    return (
        <div>
            <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text-muted)' }}>
                {label}
            </p>
            <div className="flex items-center gap-2">
                <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {value}
                </p>
                {status === 'success' && (
                    <CheckCircle className="w-4 h-4" style={{ color: '#059669' }} />
                )}
            </div>
        </div>
    );
}
