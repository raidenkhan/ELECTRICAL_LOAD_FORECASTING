'use client';

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { api } from '../lib/api';
import { getToken, setToken, removeToken, User, AuthResponse } from '../lib/auth';
import { useRouter } from 'next/navigation';

interface AuthContextType {
    user: User | null;
    loading: boolean;
    login: (email: string, password: string) => Promise<void>;
    signup: (email: string, password: string, fullName: string) => Promise<void>;
    logout: () => void;
    isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        const initAuth = async () => {
            const token = getToken();
            if (token) {
                try {
                    const { data } = await api.get<User>('/users/me');
                    setUser(data);
                } catch (error) {
                    console.error('Failed to fetch user', error);
                    removeToken();
                }
            }
            setLoading(false);
        };

        initAuth();
    }, []);

    const login = async (username: string, password: string) => {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const { data } = await api.post<AuthResponse>('/access-token', formData, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });

        setToken(data.access_token);
        const userResp = await api.get<User>('/users/me');
        setUser(userResp.data);
        router.push('/');
    };

    const signup = async (email: string, password: string, fullName: string) => {
        await api.post('/users/signup', {
            email,
            password,
            full_name: fullName,
        });
        // Auto login after signup
        await login(email, password);
    };

    const logout = () => {
        removeToken();
        setUser(null);
        router.push('/login');
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                loading,
                login,
                signup,
                logout,
                isAuthenticated: !!user,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
