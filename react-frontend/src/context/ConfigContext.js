import React, { createContext, useState, useContext, useEffect } from 'react';

const ConfigContext = createContext();

export const useConfig = () => {
  const context = useContext(ConfigContext);
  if (!context) {
    throw new Error('useConfig must be used within a ConfigProvider');
  }
  return context;
};

// Detect environment immediately
const isLocalEnvironment = () => {
  if (typeof window === 'undefined') return true;
  const hostname = window.location.hostname;
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '';
};

// Get production API URL
const PROD_API_URL = 'https://trafficflow.onrender.com';
const PROD_WS_URL = 'wss://trafficflow.onrender.com';
const LOCAL_API_URL = 'http://localhost:8000';
const LOCAL_WS_URL = 'ws://localhost:8000';

// Set initial config based on environment IMMEDIATELY
const getInitialConfig = () => {
  const isLocal = isLocalEnvironment();
  return {
    API_BASE_URL: isLocal ? LOCAL_API_URL : PROD_API_URL,
    WS_BASE_URL: isLocal ? LOCAL_WS_URL : PROD_WS_URL,
    POLL_INTERVAL: 2000,
    STREAM_POLL_INTERVAL: 1000,
    CONFIG_LOADED: true
  };
};

export const ConfigProvider = ({ children }) => {
  const [config] = useState(getInitialConfig);

  useEffect(() => {
    console.log('🌐 Environment:', isLocalEnvironment() ? 'Local' : 'Production');
    console.log('🔗 API URL:', config.API_BASE_URL);
    console.log('🔗 WS URL:', config.WS_BASE_URL);
  }, [config]);

  return <ConfigContext.Provider value={config}>{children}</ConfigContext.Provider>;
};
