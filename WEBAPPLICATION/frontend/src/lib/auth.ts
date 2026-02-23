import Cookies from 'js-cookie';

const TOKEN_KEY = 'loadforecast_token';

export const setToken = (token: string) => {
  Cookies.set(TOKEN_KEY, token, { expires: 7, secure: window.location.protocol === 'https:' });
};

export const getToken = (): string | undefined => {
  return Cookies.get(TOKEN_KEY);
};

export const removeToken = () => {
  Cookies.remove(TOKEN_KEY);
};

export interface User {
  id: number;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}
