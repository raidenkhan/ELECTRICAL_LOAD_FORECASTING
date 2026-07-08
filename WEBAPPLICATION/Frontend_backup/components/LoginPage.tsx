import { useState } from 'react';
import { Zap, User, Briefcase, Shield, BarChart3 } from 'lucide-react';

interface LoginPageProps {
  onLogin: (role: string, name: string) => void;
}

type UserRole = 'operator' | 'analyst' | 'planner' | 'admin';

const roles = [
  { id: 'operator' as UserRole, label: 'Operator', icon: Zap, description: 'Real-time monitoring and operations' },
  { id: 'analyst' as UserRole, label: 'Analyst', icon: BarChart3, description: 'Model performance and analytics' },
  { id: 'planner' as UserRole, label: 'Planner', icon: Briefcase, description: 'Forecasting and scenario planning' },
  { id: 'admin' as UserRole, label: 'Administrator', icon: Shield, description: 'System configuration and management' }
];

export function LoginPage({ onLogin }: LoginPageProps) {
  const [activeTab, setActiveTab] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [selectedRole, setSelectedRole] = useState<UserRole>('operator');
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simple validation - in production would verify credentials
    if (email && password) {
      if (activeTab === 'signup' && password !== confirmPassword) {
        alert('Passwords do not match');
        return;
      }
      if (activeTab === 'signup' && !fullName) {
        alert('Please enter your full name');
        return;
      }
      const name = activeTab === 'signup' ? fullName : email.split('@')[0];
      onLogin(selectedRole, name);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="w-full max-w-2xl">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Zap className="w-10 h-10" style={{ color: 'var(--primary-blue)' }} />
            <h1 className="text-2xl" style={{ 
              fontWeight: 'var(--font-weight-bold)', 
              color: 'var(--primary-blue)' 
            }}>
              Grid Forecast Pro
            </h1>
          </div>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Production Load Forecasting Decision Support System
          </p>
        </div>

        {/* Login Card */}
        <div 
          className="rounded-lg p-8"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
          }}
        >
          {/* Tabs */}
          <div className="flex mb-6 border-b" style={{ borderColor: 'var(--border-default)' }}>
            <button
              onClick={() => setActiveTab('login')}
              className="flex-1 pb-3 text-sm transition-colors relative"
              style={{
                fontWeight: 'var(--font-weight-medium)',
                color: activeTab === 'login' ? 'var(--primary-blue)' : 'var(--text-muted)'
              }}
            >
              Login
              {activeTab === 'login' && (
                <div 
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ backgroundColor: 'var(--primary-blue)' }}
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab('signup')}
              className="flex-1 pb-3 text-sm transition-colors relative"
              style={{
                fontWeight: 'var(--font-weight-medium)',
                color: activeTab === 'signup' ? 'var(--primary-blue)' : 'var(--text-muted)'
              }}
            >
              Sign Up
              {activeTab === 'signup' && (
                <div 
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ backgroundColor: 'var(--primary-blue)' }}
                />
              )}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Full Name Field (Sign Up only) */}
            {activeTab === 'signup' && (
              <div>
                <label 
                  htmlFor="fullName" 
                  className="block mb-2 text-sm"
                  style={{ 
                    fontWeight: 'var(--font-weight-medium)',
                    color: 'var(--text-secondary)'
                  }}
                >
                  Full Name
                </label>
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="John Doe"
                  className="w-full px-4 py-3 rounded-md transition-all duration-200"
                  style={{
                    border: '1px solid var(--border-default)',
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: 'var(--text-sm)'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = 'var(--primary-blue)';
                    e.target.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'var(--border-default)';
                    e.target.style.boxShadow = 'none';
                  }}
                  required
                />
              </div>
            )}

            {/* Email Field */}
            <div>
              <label 
                htmlFor="email" 
                className="block mb-2 text-sm"
                style={{ 
                  fontWeight: 'var(--font-weight-medium)',
                  color: 'var(--text-secondary)'
                }}
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@gridforecast.com"
                className="w-full px-4 py-3 rounded-md transition-all duration-200"
                style={{
                  border: '1px solid var(--border-default)',
                  backgroundColor: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: 'var(--text-sm)'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = 'var(--primary-blue)';
                  e.target.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'var(--border-default)';
                  e.target.style.boxShadow = 'none';
                }}
                required
              />
            </div>

            {/* Role Selection */}
            <div>
              <label 
                className="block mb-3 text-sm"
                style={{ 
                  fontWeight: 'var(--font-weight-medium)',
                  color: 'var(--text-secondary)'
                }}
              >
                Select Your Role
              </label>
              <div className="grid grid-cols-2 gap-3">
                {roles.map((role) => {
                  const Icon = role.icon;
                  const isSelected = selectedRole === role.id;
                  
                  return (
                    <button
                      key={role.id}
                      type="button"
                      onClick={() => setSelectedRole(role.id)}
                      className="p-4 rounded-lg text-left transition-all duration-200"
                      style={{
                        border: `2px solid ${isSelected ? 'var(--primary-blue)' : 'var(--border-default)'}`,
                        backgroundColor: isSelected ? '#EFF6FF' : 'var(--bg-secondary)',
                        cursor: 'pointer'
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) {
                          e.currentTarget.style.borderColor = '#CBD5E1';
                          e.currentTarget.style.backgroundColor = '#F8FAFC';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) {
                          e.currentTarget.style.borderColor = 'var(--border-default)';
                          e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
                        }
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Icon 
                          className="w-5 h-5" 
                          style={{ color: isSelected ? 'var(--primary-blue)' : 'var(--text-muted)' }} 
                        />
                        <span 
                          className="text-sm"
                          style={{ 
                            fontWeight: 'var(--font-weight-semibold)',
                            color: isSelected ? 'var(--primary-blue)' : 'var(--text-primary)'
                          }}
                        >
                          {role.label}
                        </span>
                      </div>
                      <p 
                        className="text-xs"
                        style={{ color: isSelected ? '#1E40AF' : 'var(--text-muted)' }}
                      >
                        {role.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label 
                htmlFor="password" 
                className="block mb-2 text-sm"
                style={{ 
                  fontWeight: 'var(--font-weight-medium)',
                  color: 'var(--text-secondary)'
                }}
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full px-4 py-3 rounded-md transition-all duration-200"
                style={{
                  border: '1px solid var(--border-default)',
                  backgroundColor: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: 'var(--text-sm)'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = 'var(--primary-blue)';
                  e.target.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'var(--border-default)';
                  e.target.style.boxShadow = 'none';
                }}
                required
              />
            </div>

            {/* Confirm Password Field (Sign Up only) */}
            {activeTab === 'signup' && (
              <div>
                <label 
                  htmlFor="confirmPassword" 
                  className="block mb-2 text-sm"
                  style={{ 
                    fontWeight: 'var(--font-weight-medium)',
                    color: 'var(--text-secondary)'
                  }}
                >
                  Confirm Password
                </label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter your password"
                  className="w-full px-4 py-3 rounded-md transition-all duration-200"
                  style={{
                    border: '1px solid var(--border-default)',
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: 'var(--text-sm)'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = 'var(--primary-blue)';
                    e.target.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'var(--border-default)';
                    e.target.style.boxShadow = 'none';
                  }}
                  required
                />
              </div>
            )}

            {/* Remember Me (Login only) */}
            {activeTab === 'login' && (
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember"
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="w-4 h-4 rounded cursor-pointer"
                    style={{
                      accentColor: 'var(--primary-blue)'
                    }}
                  />
                  <label 
                    htmlFor="remember" 
                    className="ml-2 text-sm cursor-pointer"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Remember me
                  </label>
                </div>
                <a 
                  href="#" 
                  className="text-sm transition-colors"
                  style={{ color: 'var(--primary-blue)' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.textDecoration = 'underline';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.textDecoration = 'none';
                  }}
                >
                  Forgot password?
                </a>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full py-3 rounded-md transition-colors duration-200"
              style={{
                backgroundColor: 'var(--primary-blue)',
                color: 'white',
                fontSize: 'var(--text-sm)',
                fontWeight: 'var(--font-weight-medium)',
                marginTop: '24px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#1D4ED8';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--primary-blue)';
              }}
            >
              {activeTab === 'login' ? 'Sign In' : 'Create Account'}
            </button>

            {/* Additional Info */}
            {activeTab === 'signup' && (
              <p className="text-xs text-center mt-4" style={{ color: 'var(--text-muted)' }}>
                By signing up, you agree to our Terms of Service and Privacy Policy
              </p>
            )}
          </form>
        </div>

        {/* Additional Help */}
        <div className="text-center mt-6">
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Need help? Contact{' '}
            <a 
              href="mailto:support@gridforecast.com"
              style={{ color: 'var(--primary-blue)' }}
              className="transition-colors"
              onMouseEnter={(e) => {
                e.currentTarget.style.textDecoration = 'underline';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.textDecoration = 'none';
              }}
            >
              support@gridforecast.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}