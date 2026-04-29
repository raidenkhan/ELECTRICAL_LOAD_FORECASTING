'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import { useTheme } from 'next-themes';
import {
  Zap, User, Briefcase, Shield, BarChart3, MapPin, ChevronDown, Building2,
  Mail, Lock, Globe, Activity, Cpu, Database, Sun, Moon, ArrowRight,
  Server, Network, BarChart, Settings, CheckCircle2, FlaskConical,
  GitMerge, Layers, Code2, Terminal, Webhook, Key, Clock, TrendingUp
} from 'lucide-react';
import { toast } from 'sonner';

interface LoginPageProps {
  onSignIn: (email: string, password: string) => Promise<void>;
  onSignUp: (email: string, password: string, fullName: string, role: string, region?: string, organization?: string) => Promise<void>;
}

type UserRole = 'operator' | 'analyst' | 'planner' | 'admin';

const roles = [
  { id: 'operator' as UserRole, label: 'Operator', icon: Zap, description: 'Real-time monitoring' },
  { id: 'analyst' as UserRole, label: 'Analyst', icon: BarChart3, description: 'Performance analytics' },
  { id: 'planner' as UserRole, label: 'Planner', icon: Briefcase, description: 'Strategic planning' },
  { id: 'admin' as UserRole, label: 'Administrator', icon: Shield, description: 'System management' }
];

const regions = [
  { id: 'accra', name: 'Greater Accra Region', code: 'GAR', capacity: '1,200 MW' },
  { id: 'ashanti', name: 'Ashanti Region', code: 'ASH', capacity: '850 MW' },
  { id: 'eastern', name: 'Eastern Region', code: 'EAS', capacity: '620 MW' },
  { id: 'western', name: 'Western Region', code: 'WES', capacity: '740 MW' },
  { id: 'northern', name: 'Northern Region', code: 'NOR', capacity: '380 MW' },
  { id: 'central', name: 'Central Region', code: 'CEN', capacity: '450 MW' }
];

export function LoginPage({ onSignIn, onSignUp }: LoginPageProps) {
  const [activeTab, setActiveTab] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [organization, setOrganization] = useState('');
  const [selectedRole, setSelectedRole] = useState<UserRole>('operator');
  const [selectedRegion, setSelectedRegion] = useState(regions[0].id);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Background animation signal flow
  const [signals, setSignals] = useState<{ id: number; x: number; y: number }[]>([]);
  useEffect(() => {
    const interval = setInterval(() => {
      setSignals(prev => [...prev.slice(-10), { id: Date.now(), x: Math.random() * 100, y: Math.random() * 100 }]);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (activeTab === 'signup') {
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          toast.error('Passwords do not match');
          setLoading(false);
          return;
        }
        await onSignUp(email, password, fullName, selectedRole, selectedRegion, organization);
        toast.success('Account created successfully');
      } else {
        await onSignIn(email, password);
        toast.success('Logged in successfully');
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Authentication failed. Please check your credentials.';
      setError(msg);
      toast.error(msg === 'Unauthorized' ? 'Unauthorized' : msg);
    } finally {
      setLoading(false);
    }
  };

  if (!mounted) return null;

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30 selection:text-primary transition-colors duration-500 overflow-x-hidden"
      style={{ fontFamily: 'var(--font-space-grotesk)' }}>

      {/* 1. STICKY NAVIGATION */}
      <StickyNavbar theme={theme} setTheme={setTheme} />

      {/* 2. SCADA TICKER */}
      <ScadaTicker />

      {/* 3. HERO SECTION (Redesigned 65/35 with scroll triggers) */}
      <section className="relative min-h-screen flex items-center justify-center pt-24 overflow-hidden border-b border-border">
        {/* Dynamic Background */}
        <TechnicalBackground signals={signals} />

        <div className="w-full max-w-[1600px] px-8 lg:px-24 flex flex-col lg:flex-row items-center lg:items-stretch gap-24 py-24">

          {/* BRANDING COLUMN */}
          <motion.div
            className="flex-[1.8] flex flex-col justify-center space-y-12"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div className="space-y-6">
              <motion.div className="inline-flex items-center gap-2 px-4 py-1 bg-primary/10 border border-primary/20 text-primary text-[10px] font-black tracking-[0.4em] uppercase"
                initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.5 }}>
                <Activity className="w-3 h-3" /> System Operational // Node_Gh_01
              </motion.div>

              <h1 className="text-7xl lg:text-8xl font-black tracking-tighter leading-[0.9] text-foreground">
                GRID FORECAST <span className="text-primary italic">PRO</span>
              </h1>
              <p className="text-xl lg:text-2xl text-muted-foreground font-light max-w-2xl leading-relaxed">
                Empowering Ghana's national grid with <span className="text-primary font-medium">high-fidelity load architectural telemetry</span> and recursive horizon prediction layers.
              </p>
            </div>

            <div className="flex flex-wrap gap-6">
              <button
                onClick={() => document.getElementById('technology')?.scrollIntoView({ behavior: 'smooth' })}
                className="h-16 px-10 bg-primary text-background font-black text-sm tracking-[0.2em] uppercase transition-all hover:scale-105 active:scale-95 flex items-center gap-4">
                Explore Technology <ArrowRight className="w-5 h-5" />
              </button>
              <button
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                className="h-16 px-10 border border-border bg-card/50 backdrop-blur-md text-foreground font-black text-sm tracking-[0.2em] uppercase transition-all hover:border-primary/50">
                System Docs
              </button>
            </div>

            <div className="grid grid-cols-3 gap-8 pt-12 border-t border-border/50">
              <ShowcaseStat label="GRID CAPACITY" value="4.2 GW" icon={Activity} />
              <ShowcaseStat label="SYSTEM UPTIME" value="99.2%" icon={Shield} />
              <ShowcaseStat label="GRANULARITY" value="15 MIN" icon={Cpu} />
            </div>
          </motion.div>

          {/* AUTH CARD COLUMN */}
          <motion.div
            className="flex-shrink-0 flex-grow-0 basis-full lg:basis-[440px] flex items-center justify-center lg:justify-end"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div className="w-full max-w-[440px] bg-card/80 backdrop-blur-2xl border border-border p-8 lg:p-12 shadow-2xl relative" style={{ width: '100%' }}>
              <div className="absolute -top-1 -left-1 w-6 h-6 border-t-2 border-l-2 border-primary" />
              <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-2 border-r-2 border-primary" />

              <div className="flex bg-muted p-1 border border-border mb-10">
                <AuthTab active={activeTab === 'login'} onClick={() => setActiveTab('login')} label="SIGN IN" />
                <AuthTab active={activeTab === 'signup'} onClick={() => setActiveTab('signup')} label="GET ACCESS" />
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    {activeTab === 'signup' ? (
                      <div className="space-y-6 overflow-y-auto max-h-[400px] pr-2 scrollbar-thin scrollbar-thumb-border">
                        <InputField label="00 // FULL NAME" placeholder="Project Engineer" value={fullName} onChange={setFullName} />
                        <InputField label="00 // ORGANIZATION" placeholder="Ghana Grid Company (GRIDCo)" value={organization} onChange={setOrganization} />

                        <div className="space-y-3">
                          <label className="text-[10px] font-bold tracking-[0.3em] text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                            00 // OPERATING REGION
                          </label>
                          <div className="relative">
                            <select
                              value={selectedRegion}
                              onChange={(e) => setSelectedRegion(e.target.value)}
                              className="w-full h-14 bg-muted border border-border px-6 text-sm text-foreground focus:border-primary focus:ring-0 appearance-none outline-none transition-all"
                              style={{ fontFamily: 'var(--font-geist-mono)', borderRadius: 0 }}
                            >
                              {regions.map(r => (
                                <option key={r.id} value={r.id}>{r.name} ({r.capacity})</option>
                              ))}
                            </select>
                            <ChevronDown className="absolute right-5 top-5 w-4 h-4 text-muted-foreground pointer-events-none" />
                          </div>
                        </div>

                        <div className="space-y-3">
                          <label className="text-[10px] font-bold tracking-[0.3em] text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                            00 // YOUR ROLE
                          </label>
                          <div className="grid grid-cols-2 gap-3">
                            {roles.map((role) => {
                              const Icon = role.icon;
                              const isSelected = selectedRole === role.id;
                              return (
                                <button
                                  key={role.id}
                                  type="button"
                                  onClick={() => setSelectedRole(role.id as UserRole)}
                                  className={`p-4 border text-left transition-all ${isSelected ? 'border-primary bg-primary/5' : 'border-border bg-muted hover:border-muted-foreground'}`}
                                >
                                  <Icon className={`w-4 h-4 mb-2 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
                                  <div className={`text-[10px] font-black tracking-widest uppercase ${isSelected ? 'text-foreground' : 'text-muted-foreground'}`} style={{ fontFamily: 'var(--font-geist-mono)' }}>
                                    {role.label}
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        <InputField label="01 // EMAIL" placeholder="user@ghana-grid.com" value={email} onChange={setEmail} type="email" />
                        <InputField label="02 // PASSWORD" placeholder="••••••••••••" value={password} onChange={setPassword} type="password" />
                        <InputField label="03 // CONFIRM" placeholder="••••••••••••" value={confirmPassword} onChange={setConfirmPassword} type="password" />
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <InputField label="01 // EMAIL" placeholder="user@ghana-grid.com" value={email} onChange={setEmail} type="email" />
                        <InputField label="02 // PASSWORD" placeholder="••••••••••••" value={password} onChange={setPassword} type="password" />

                        <div className="flex items-center justify-between pt-2">
                          <label className="flex items-center gap-3 cursor-pointer group">
                            <div className="w-4 h-4 border border-border group-hover:border-primary transition-colors bg-muted flex items-center justify-center">
                              <div className="w-2 h-2 bg-primary scale-0 group-active:scale-100 transition-transform" />
                            </div>
                            <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                              Persistent
                            </span>
                          </label>
                          <button type="button" className="text-[10px] font-bold tracking-widest text-primary uppercase hover:underline" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                            Recovery
                          </button>
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>

                <motion.button
                  type="submit"
                  disabled={loading}
                  className="w-full h-16 bg-primary text-background text-[12px] font-black tracking-[0.4em] relative group overflow-hidden mt-8"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  style={{ fontFamily: 'var(--font-geist-mono)' }}
                >
                  {loading ? 'INITIALIZING...' : (activeTab === 'login' ? 'INITIALIZE SESSION' : 'CREATE ACCOUNT')}
                </motion.button>
              </form>
            </div>
          </motion.div>
        </div>
      </section>

      {/* 4. FEATURES SECTION */}
      <section id="features" className="py-24 border-b border-border bg-muted/30">
        <div className="max-w-[1600px] mx-auto px-8 lg:px-24">
          <div className="flex flex-col lg:flex-row gap-24">
            <div className="lg:w-1/3 space-y-8">
              <h2 className="text-5xl font-black tracking-tighter leading-none">
                GRID-SCALE <br /> <span className="text-primary italic">INTELLIGENCE</span>
              </h2>
              <p className="text-lg text-muted-foreground">
                Our forecasting engine is built on mission-critical architecture designed for high-availability national grid operations.
              </p>
              <div className="space-y-4 pt-8">
                <FeatureSmall icon={Server} title="Distributed SCADA Interface" />
                <FeatureSmall icon={Network} title="Neural-Regime Switching" />
                <FeatureSmall icon={Database} title="10-Year Vector Database" />
              </div>
            </div>

            <div className="lg:w-2/3 grid grid-cols-1 md:grid-cols-2 gap-6">
              <FeatureCard
                icon={Activity}
                title="Real-Time Telemetry"
                desc="15-minute granularity processing from over 240 grid nodes across Ghana's high-voltage transmission network."
              />
              <FeatureCard
                icon={Cpu}
                title="SOTA Architectures"
                desc="Hybrid LSTM-Transformer ensemble models achieving 95%+ accuracy in complex tropical demand profiles."
              />
              <FeatureCard
                icon={Globe}
                title="Regional Attribution"
                desc="Granular demand forecasting down to the substation level, enabling precise load shedding mitigation."
              />
              <FeatureCard
                icon={Shield}
                title="Critical Availability"
                desc="Redundant compute nodes ensuring 99.9% uptime for national control room decision making."
              />
            </div>
          </div>
        </div>
      </section>

      {/* 5. ARCHITECTURE VISUALIZATION */}
      <section id="technology" className="py-32 overflow-hidden bg-background">
        <div className="max-w-[1600px] mx-auto px-8 lg:px-24 flex flex-col items-center text-center space-y-24">
          <div className="max-w-3xl space-y-8">
            <h2 className="text-6xl font-black tracking-tighter italic text-primary">RECURSIVE HORIZON ROUTING</h2>
            <p className="text-xl text-muted-foreground font-light leading-relaxed">
              Our proprietary <span className="text-foreground font-medium">multi-vector pipeline</span> ensures that whether the forecast is for the next 15 minutes or the next 7 days, the most accurate neural regime is selected dynamically.
            </p>
          </div>

          {/* Interactive Architecture Map/Diagram Mockup */}
          <div className="relative w-full max-w-5xl aspect-video bg-muted border border-border group">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="grid grid-cols-4 gap-12 relative w-full px-24">
                <ArchNode icon={Database} label="DATA INGEST" active />
                <ArchNode icon={Network} label="FEATURE ENGIN" active />
                <ArchNode icon={Cpu} label="MODEL INFER" />
                <ArchNode icon={BarChart} label="GRID OUTPUT" />

                {/* Connecting Lines (Simplified) */}
                <div className="absolute top-1/2 left-0 w-full h-[1px] bg-border z-0" />
                <motion.div
                  className="absolute top-1/2 left-0 h-[2px] bg-primary z-0"
                  initial={{ width: 0 }}
                  whileInView={{ width: '100%' }}
                  viewport={{ once: true }}
                  transition={{ duration: 2, ease: "easeInOut" }}
                />
              </div>
            </div>
            <div className="absolute bottom-8 right-8 text-[10px] font-black text-muted-foreground tracking-widest uppercase bg-background px-3 py-1 border border-border">
              Internal Architecture Ver 4.2.0-Alpha
            </div>
          </div>
        </div>
      </section>

      {/* 6. KALMAN FUSION TECHNOLOGY SECTION */}
      <section id="kalman" className="py-32 border-b border-border bg-muted/20">
        <div className="max-w-[1600px] mx-auto px-8 lg:px-24 space-y-24">
          <div className="flex flex-col lg:flex-row gap-16 items-start">
            <div className="lg:w-2/5 space-y-8 lg:sticky lg:top-32">
              <div className="inline-flex items-center gap-2 px-4 py-1 bg-primary/10 border border-primary/20 text-primary text-xs font-black tracking-[0.4em] uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                <GitMerge className="w-3 h-3" /> Core Algorithm
              </div>
              <h2 className="text-5xl lg:text-6xl font-black tracking-tighter leading-none">
                KALMAN<br /><span className="text-primary italic">FUSION</span><br />ENGINE
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Our proprietary <span className="text-foreground font-semibold">Bayesian state-space fusion layer</span> dynamically weights predictions from the Autoformer transformer and LightGBM gradient booster — selecting the optimal blend based on real-time uncertainty estimates.
              </p>
              <div className="space-y-4 pt-4">
                <KalmanStat label="Variance Reduction" value="38%" />
                <KalmanStat label="Ensemble MAPE" value="1.2%" />
                <KalmanStat label="Fusion Latency" value="<4ms" />
              </div>
            </div>

            <div className="lg:w-3/5 space-y-6">
              {/* Pipeline Steps */}
              {[
                {
                  step: '01',
                  icon: Database,
                  title: 'Historical Context Window',
                  desc: 'The engine ingests the last 672 timesteps (7 days × 96 steps/day at 15-min granularity) from the ValidatedData store, constructing a rich temporal context for both models.',
                  code: 'context_window = df.tail(672)  # 7-day lookback'
                },
                {
                  step: '02',
                  icon: Layers,
                  title: 'Dual-Model Inference',
                  desc: 'Autoformer (transformer-based) captures long-range seasonal dependencies. LightGBM handles non-linear feature interactions. Both produce independent point forecasts and variance estimates.',
                  code: 'pred_af, var_af = autoformer.predict(x_enc)\npred_lgb, var_lgb = lightgbm.predict(x_feat)'
                },
                {
                  step: '03',
                  icon: GitMerge,
                  title: 'Inverse-Variance Weighting',
                  desc: 'The Kalman layer computes optimal weights inversely proportional to each model\'s prediction variance. Lower uncertainty = higher weight. This is equivalent to a one-step Kalman update.',
                  code: 'w_af  = 1/var_af\nw_lgb = 1/var_lgb\nfused = (w_af*pred_af + w_lgb*pred_lgb) / (w_af + w_lgb)'
                },
                {
                  step: '04',
                  icon: TrendingUp,
                  title: 'Uncertainty Quantification',
                  desc: 'The fused variance is propagated to produce calibrated P10/P90 prediction intervals, enabling grid operators to plan for best-case and worst-case demand scenarios.',
                  code: 'fused_var = 1 / (w_af + w_lgb)\np10 = fused - 1.28 * sqrt(fused_var)\np90 = fused + 1.28 * sqrt(fused_var)'
                }
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="flex gap-8 p-8 bg-card border border-border hover:border-primary/50 transition-all group"
                  initial={{ opacity: 0, x: 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1, duration: 0.5 }}
                >
                  <div className="flex-shrink-0">
                    <div className="w-14 h-14 bg-primary/10 border border-primary/20 flex items-center justify-center group-hover:bg-primary group-hover:border-primary transition-all">
                      <item.icon className="w-6 h-6 text-primary group-hover:text-background transition-colors" />
                    </div>
                  </div>
                  <div className="space-y-3 flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-base font-black text-primary tracking-widest" style={{ fontFamily: 'var(--font-geist-mono)' }}>{item.step}</span>
                      <h3 className="text-2xl font-black tracking-tight">{item.title}</h3>
                    </div>
                    <p className="text-lg text-muted-foreground leading-relaxed">{item.desc}</p>
                    <div className="bg-muted border border-border p-4 mt-2">
                      <pre className="text-base text-primary leading-relaxed overflow-x-auto" style={{ fontFamily: 'var(--font-geist-mono)' }}>{item.code}</pre>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 7. API REFERENCE SECTION */}
      <section id="api" className="py-32 border-b border-border bg-background">
        <div className="max-w-[1600px] mx-auto px-8 lg:px-24 space-y-16">
          <div className="flex flex-col lg:flex-row gap-8 items-end justify-between">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1 bg-primary/10 border border-primary/20 text-primary text-xs font-black tracking-[0.4em] uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
                <Webhook className="w-3 h-3" /> REST API v1
              </div>
              <h2 className="text-5xl lg:text-6xl font-black tracking-tighter leading-none">
                API<br /><span className="text-primary italic">REFERENCE</span>
              </h2>
              <p className="text-lg text-muted-foreground max-w-xl">
                All endpoints are served over HTTPS with JWT bearer authentication. Base URL: <code className="text-primary bg-muted px-2 py-0.5 text-sm" style={{ fontFamily: 'var(--font-geist-mono)' }}>https://api.gridforecast.gh/v1</code>
              </p>
            </div>
            <div className="flex items-center gap-4 p-4 bg-muted border border-border">
              <Key className="w-5 h-5 text-primary flex-shrink-0" />
              <div>
                <div className="text-xs font-black tracking-widest text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>Authorization Header</div>
                <code className="text-base text-foreground" style={{ fontFamily: 'var(--font-geist-mono)' }}>Bearer &lt;JWT_TOKEN&gt;</code>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ApiEndpointCard
              method="POST"
              path="/forecast/stlf"
              description="Generate a Short-Term Load Forecast (STLF) using the Kalman-fused Autoformer + LightGBM ensemble. Returns 24h forecast with P10/P90 intervals and operating regime distribution."
              request={`{
  "horizon_hours": 24,
  "model_type": "stlf"
}`}
              response={`{
  "forecast_id": "stlf_202602181200",
  "model_type": "stlf",
  "forecast_mw": [98.4, 101.2, ...],
  "p10": [91.2, 94.1, ...],
  "p90": [105.6, 108.3, ...],
  "regime_distribution": [
    {"hour": "00-04", "regime0": 60.0,
     "regime1": 30.0, "regime2": 10.0}
  ]
}`}
            />
            <ApiEndpointCard
              method="POST"
              path="/forecast/ltlf"
              description="Generate a Long-Term Load Forecast (LTLF) using recursive LightGBM quantile regressors. Supports up to 30-day horizons for capacity planning."
              request={`{
  "horizon_hours": 720,
  "model_type": "ltlf"
}`}
              response={`{
  "forecast_id": "ltlf_20260218",
  "model_type": "ltlf",
  "forecast_mw": [97.1, 99.3, ...],
  "p10": [88.4, 90.1, ...],
  "p90": [106.0, 108.5, ...]
}`}
            />
            <ApiEndpointCard
              method="POST"
              path="/forecast/simulate"
              description="Run a what-if scenario simulation. Adjust temperature, grid inflow, and industrial load offsets to model demand under hypothetical conditions."
              request={`{
  "horizon_hours": 24,
  "temp_offset": 3.5,
  "inflow_offset_pct": -10.0,
  "industrial_load_offset_pct": 5.0
}`}
              response={`{
  "forecast_id": "sim_202602181200",
  "model_type": "simulation",
  "forecast_mw": [104.2, 107.8, ...],
  "metadata": {
    "offsets": {"temp": 3.5,
      "inflow": -10.0, "industrial": 5.0}
  }
}`}
            />
            <ApiEndpointCard
              method="GET"
              path="/data/latest"
              description="Retrieve the most recent validated telemetry records from the SCADA ingestion pipeline. Supports pagination via the limit query parameter."
              request={`GET /data/latest?limit=96
Authorization: Bearer <token>`}
              response={`[
  {
    "timestamp": "2026-02-18T12:00:00Z",
    "total_load_mw": 102.4,
    "frequency_hz": 49.98,
    "voltage_kv": 161.2,
    "line1_mw": 38.1,
    "line2_mw": 34.6
  }, ...
]`}
            />
            <ApiEndpointCard
              method="GET"
              path="/explain/shap"
              description="Retrieve SHAP (SHapley Additive exPlanations) feature importance values for the latest STLF ensemble inference, enabling model explainability audits."
              request={`GET /explain/shap
Authorization: Bearer <token>`}
              response={`{
  "features": ["Lag_96_Load",
    "Rolling_Mean_24h", "Hour_Sin"],
  "values": [45.2, 28.5, -12.4],
  "base_value": 83.6
}`}
            />
            <ApiEndpointCard
              method="GET"
              path="/models/metrics"
              description="Retrieve benchmark performance metrics (MAE, RMSE, MAPE) for each forecast horizon. Used for model health monitoring and SLA reporting."
              request={`GET /models/metrics
Authorization: Bearer <token>`}
              response={`[
  {"horizon": "STLF (24h)",
   "mae": 15.4, "rmse": 22.1,
   "mape": 1.2, "sample_count": 1440},
  {"horizon": "LTLF (720h)",
   "mae": 85.2, "rmse": 112.7,
   "mape": 6.8, "sample_count": 120}
]`}
            />
          </div>
        </div>
      </section>
      <footer className="py-24 border-t border-border bg-muted/50 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-8 lg:px-24 grid grid-cols-1 md:grid-cols-4 gap-16">
          <div className="col-span-1 md:col-span-2 space-y-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary flex items-center justify-center">
                <span className="text-xl font-black text-background">G</span>
              </div>
              <h3 className="text-2xl font-black tracking-tighter">GRID FORECAST <span className="text-primary italic">PRO</span></h3>
            </div>
            <p className="text-muted-foreground text-sm leading-relaxed max-w-md">
              Grid Forecast Pro is a state-of-the-art electrical telemetry and demand prediction platform developed for the next generation of industrial grid management.
            </p>
            <div className="pt-8 text-[9px] font-bold tracking-[0.2em] text-muted-foreground/50 uppercase">
              GHANA GRID TECHNOLOGY INITIATIVE // 2026 // ALL RIGHTS RESERVED
            </div>
          </div>

          <div className="space-y-6">
            <h4 className="text-[10px] font-black tracking-[0.4em] text-foreground uppercase border-b border-primary/20 pb-2">Ecosystem</h4>
            <FooterLink label="Grid Monitoring" />
            <FooterLink label="Demand Analytics" />
            <FooterLink label="Regional Planning" />
            <FooterLink label="API Access" />
          </div>

          <div className="space-y-6">
            <h4 className="text-[10px] font-black tracking-[0.4em] text-foreground uppercase border-b border-primary/20 pb-2">Support</h4>
            <FooterLink label="Operation Manuals" />
            <FooterLink label="System Status" />
            <FooterLink label="Security Policy" />
            <FooterLink label="Contact Terminal" />
          </div>
        </div>
      </footer>
    </div>
  );
}

// SUB-COMPONENTS

function StickyNavbar({ theme, setTheme }: { theme: any; setTheme: any }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 w-full z-[100] transition-all duration-300 ${scrolled ? 'h-20 bg-background/80 backdrop-blur-xl border-b border-border' : 'h-24 bg-transparent'}`}>
      <div className="max-w-[1600px] h-full mx-auto px-8 lg:px-24 flex items-center justify-between">
        <div className="flex items-center gap-4 group cursor-pointer">
          <div className="w-10 h-10 bg-primary flex items-center justify-center transition-transform group-hover:rotate-[15deg]">
            <span className="text-lg font-black text-background">G</span>
          </div>
          <span className="text-xl font-black tracking-tighter text-foreground hidden sm:block">GRID FORECAST <span className="text-primary italic">PRO</span></span>
        </div>

        <div className="hidden lg:flex items-center gap-12">
          <NavLink label="Technology" href="#technology" />
          <NavLink label="Features" href="#features" />
          <NavLink label="Kalman Fusion" href="#kalman" />
          <NavLink label="API" href="#api" />
        </div>

        <div className="flex items-center gap-6">
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-3 bg-muted border border-border text-foreground hover:border-primary transition-all active:scale-95"
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>

          <button className="h-12 px-6 bg-primary text-background font-black text-[10px] tracking-widest uppercase transition-all hover:scale-105 active:scale-95">
            Initialize Access
          </button>
        </div>
      </div>
    </nav>
  );
}

function ScadaTicker() {
  const items = [
    "NODE_ACCRA_CENTRAL: 231.2V [OK]", "GRID_FREQ: 49.98Hz [STABLE]", "PEAK_DEMAND: 4,212MW",
    "RESERVE_CAP: 840MW", "VOLTAGE_STABILITY: 99.1%", "LOAD_REGIME: TRANSITION",
    "TRANS_LOSSES: 3.2%", "NODE_KUMASI_NORTH: 229.4V [OK]", "SYSTEM_THREAT: NULL"
  ];

  return (
    <div className="fixed top-24 left-0 w-full bg-primary/5 border-y border-primary/20 backdrop-blur-sm h-10 z-50 flex items-center overflow-hidden">
      <motion.div
        className="flex gap-16 whitespace-nowrap px-8"
        animate={{ x: [0, -1000] }}
        transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
      >
        {[...items, ...items].map((item, i) => (
          <span key={i} className="text-[10px] font-bold text-primary tracking-widest uppercase flex items-center gap-2" style={{ fontFamily: 'var(--font-geist-mono)' }}>
            <div className="w-1.5 h-1.5 bg-primary" /> {item}
          </span>
        ))}
      </motion.div>
    </div>
  );
}

function TechnicalBackground({ signals }: { signals: any[] }) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 20 - 10,
        y: (e.clientY / window.innerHeight) * 20 - 10
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
      {/* Static texture overlay */}
      <div className="absolute inset-0 opacity-[0.03] dark:opacity-[0.02]"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?q=80&w=2069&auto=format&fit=crop')`,
          backgroundSize: 'cover', backgroundPosition: 'center', filter: 'grayscale(100%) contrast(120%)'
        }} />

      {/* Parallax Glow Blobs */}
      <motion.div
        className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/20 blur-[120px]"
        animate={{ x: mousePosition.x * -2, y: mousePosition.y * -2 }}
        transition={{ type: 'tween', ease: 'linear', duration: 0.2 }}
      />
      <motion.div
        className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-primary/10 blur-[100px]"
        animate={{ x: mousePosition.x * 3, y: mousePosition.y * 3 }}
        transition={{ type: 'tween', ease: 'linear', duration: 0.2 }}
      />
      <motion.div
        className="absolute top-[20%] right-[20%] w-[30%] h-[30%] rounded-full bg-primary/5 blur-[80px]"
        animate={{ x: mousePosition.x * -1, y: mousePosition.y * -1 }}
        transition={{ type: 'tween', ease: 'linear', duration: 0.2 }}
      />

      {/* Signal Grid */}
      <svg className="absolute inset-0 w-full h-full opacity-20">
        <AnimatePresence>
          {signals.map((signal: any) => (
            <motion.circle
              key={signal.id} cx={`${signal.x}%`} cy={`${signal.y}%`} r="1" fill="var(--lime-primary)"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: [0, 1, 0], scale: [0, 4, 0] }}
              exit={{ opacity: 0 }}
              transition={{ duration: 3, ease: "linear" }}
            />
          ))}
        </AnimatePresence>
      </svg>

      <div className="absolute inset-0 opacity-[0.05]"
        style={{ backgroundImage: 'radial-gradient(var(--lime-primary) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
    </div>
  );
}


function ArchNode({ icon: Icon, label, active = false }: any) {
  return (
    <div className="flex flex-col items-center gap-4 z-10">
      <div className={`w-20 h-20 flex items-center justify-center transition-all duration-700
                      ${active ? 'bg-primary border-primary shadow-[0_0_30px_rgba(204,255,0,0.3)]' : 'bg-background border-border shadow-none'} border-2`}>
        <Icon className={`w-8 h-8 ${active ? 'text-background' : 'text-muted-foreground'}`} />
      </div>
      <span className={`text-[10px] font-black tracking-widest uppercase ${active ? 'text-foreground' : 'text-muted-foreground'}`}>{label}</span>
    </div>
  );
}

function ShowcaseStat({ label, value, icon: Icon }: { label: string; value: string; icon: any }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-primary">
        <Icon className="w-5 h-5" />
        <span className="text-[10px] font-black tracking-[0.3em] uppercase">{label}</span>
      </div>
      <div className="text-4xl font-black text-foreground" style={{ fontFamily: 'var(--font-space-grotesk)' }}>{value}</div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, desc }: any) {
  return (
    <div className="p-10 bg-card border border-border transition-all hover:border-primary group">
      <Icon className="w-10 h-10 text-primary mb-8" />
      <h3 className="text-2xl font-black tracking-tighter mb-4 group-hover:text-primary transition-colors">{title}</h3>
      <p className="text-muted-foreground leading-relaxed text-sm">{desc}</p>
    </div>
  );
}

function FeatureSmall({ icon: Icon, title }: any) {
  return (
    <div className="flex items-center gap-4 group cursor-pointer">
      <div className="w-8 h-8 bg-muted border border-border flex items-center justify-center group-hover:bg-primary group-hover:border-primary transition-all">
        <Icon className="w-4 h-4 text-muted-foreground group-hover:text-background" />
      </div>
      <span className="text-xs font-black tracking-widest text-foreground uppercase">{title}</span>
    </div>
  );
}

function NavLink({ label, href }: { label: string; href: string }) {
  return (
    <a href={href} className="text-[11px] font-black tracking-[0.3em] text-muted-foreground uppercase hover:text-primary transition-colors">
      {label}
    </a>
  );
}

function FooterLink({ label }: { label: string }) {
  return (
    <a href="#" className="block text-[11px] font-bold text-muted-foreground hover:text-primary transition-colors uppercase tracking-widest">
      {label}
    </a>
  );
}

function KalmanStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-border/50">
      <span className="text-base font-bold tracking-widest text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>{label}</span>
      <span className="text-3xl font-black text-primary" style={{ fontFamily: 'var(--font-space-grotesk)' }}>{value}</span>
    </div>
  );
}

function ApiEndpointCard({ method, path, description, request, response }: {
  method: 'GET' | 'POST' | 'DELETE' | 'PUT';
  path: string;
  description: string;
  request: string;
  response: string;
}) {
  const methodColors: Record<string, string> = {
    GET: 'text-sky-400 bg-sky-400/10 border-sky-400/20',
    POST: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
    DELETE: 'text-red-400 bg-red-400/10 border-red-400/20',
    PUT: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
  };
  const [open, setOpen] = useState(false);
  return (
    <div className="bg-card border border-border hover:border-primary/40 transition-all">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-start gap-4 p-6 text-left"
      >
        <span className={`flex-shrink-0 px-3 py-1 text-sm font-black tracking-widest border ${methodColors[method]}`} style={{ fontFamily: 'var(--font-geist-mono)' }}>
          {method}
        </span>
        <div className="flex-1 min-w-0">
          <div className="text-xl font-black text-foreground tracking-tight mb-1" style={{ fontFamily: 'var(--font-geist-mono)' }}>{path}</div>
          <div className="text-base text-muted-foreground leading-relaxed">{description}</div>
        </div>
        <ChevronDown className={`w-4 h-4 text-muted-foreground flex-shrink-0 mt-1 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-border grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
          <div className="p-4">
            <div className="text-sm font-black tracking-[0.3em] text-muted-foreground uppercase mb-2" style={{ fontFamily: 'var(--font-geist-mono)' }}>Request</div>
            <pre className="text-base text-sky-400 leading-relaxed overflow-x-auto" style={{ fontFamily: 'var(--font-geist-mono)' }}>{request}</pre>
          </div>
          <div className="p-4">
            <div className="text-sm font-black tracking-[0.3em] text-muted-foreground uppercase mb-2" style={{ fontFamily: 'var(--font-geist-mono)' }}>Response</div>
            <pre className="text-base text-emerald-400 leading-relaxed overflow-x-auto" style={{ fontFamily: 'var(--font-geist-mono)' }}>{response}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

function InputField({ label, placeholder, value, onChange, type = "text" }: any) {
  return (
    <div className="space-y-3">
      <label className="text-[10px] font-bold tracking-[0.3em] text-muted-foreground uppercase" style={{ fontFamily: 'var(--font-geist-mono)' }}>
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-14 bg-muted border border-border px-6 text-sm text-foreground focus:border-primary focus:ring-0 transition-all outline-none"
        style={{ fontFamily: 'var(--font-geist-mono)', borderRadius: 0 }}
        placeholder={placeholder}
      />
    </div>
  );
}

function AuthTab({ active, onClick, label }: any) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 py-4 text-[11px] font-black tracking-[0.3em] transition-all relative
                 ${active ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
      style={{ fontFamily: 'var(--font-geist-mono)' }}
    >
      {label}
      {active && (
        <motion.div
          layoutId="activeTab" className="absolute inset-0 bg-primary/5"
          transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
        />
      )}
    </button>
  );
}