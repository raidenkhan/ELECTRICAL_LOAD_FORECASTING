'use client';

import { useAuth } from '../../context/AuthContext';
import { LoginPage } from '../../components/LoginPage';

export default function LoginRoute() {
    const { login, signup } = useAuth();

    const handleSignIn = async (email: string, password: string) => {
        await login(email, password);
    };

    const handleSignUp = async (email: string, password: string, fullName: string, role: string, region?: string, organization?: string) => {
        // Note: Role, Region, and Organization are currently not supported by the backend signup endpoint
        // We strictly pass email, password, and fullName
        await signup(email, password, fullName);
    };

    return <LoginPage onSignIn={handleSignIn} onSignUp={handleSignUp} />;
}
