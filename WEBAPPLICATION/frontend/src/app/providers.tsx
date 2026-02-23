'use client';

import { AuthProvider } from '../context/AuthContext';
import { ThemeProvider as NextThemesProvider } from 'next-themes';

export function Providers({ children }: { children: React.ReactNode }) {
    return (
        <NextThemesProvider attribute="class" defaultTheme="dark" enableSystem={false}>
            <AuthProvider>
                {children}
            </AuthProvider>
        </NextThemesProvider>
    );
}
