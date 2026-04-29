'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

interface SystemContextType {
  lastSync: Date | null;
  updateSync: () => void;
  isStale: boolean;
  secondsSinceSync: number;
}

const SystemContext = createContext<SystemContextType | undefined>(undefined);

export function SystemProvider({ children }: { children: React.ReactNode }) {
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [secondsSinceSync, setSecondsSinceSync] = useState(0);

  const updateSync = useCallback(() => {
    setLastSync(new Date());
  }, []);

  useEffect(() => {
    // Initial sync on mount if needed, or wait for first data fetch
    if (!lastSync) {
        setLastSync(new Date());
    }

    const interval = setInterval(() => {
      if (lastSync) {
        const diff = Math.floor((new Date().getTime() - lastSync.getTime()) / 1000);
        setSecondsSinceSync(diff);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [lastSync]);

  const isStale = secondsSinceSync > 900; // 15 minutes = 900 seconds

  return (
    <SystemContext.Provider value={{ lastSync, updateSync, isStale, secondsSinceSync }}>
      {children}
    </SystemContext.Provider>
  );
}

export function useSystem() {
  const context = useContext(SystemContext);
  if (context === undefined) {
    throw new Error('useSystem must be used within a SystemProvider');
  }
  return context;
}
