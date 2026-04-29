'use client';

import { AuthProvider } from '../context/AuthContext';
import { SystemProvider } from '@/context/SystemContext';
import { ThemeProvider as NextThemesProvider } from 'next-themes';
import { Toaster } from '@/components/ui/sonner';

export function Providers({ children }: { children: React.ReactNode }) {
    return (
        <NextThemesProvider attribute="class" defaultTheme="dark" enableSystem={false}>
            <SystemProvider>
                <AuthProvider>
                    {children}
                    <Toaster />
                </AuthProvider>
            </SystemProvider>
        </NextThemesProvider>
    );
}
