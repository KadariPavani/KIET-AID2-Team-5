import React, { createContext, useState, useContext, useEffect } from 'react';

const ConfigContext = createContext();

export const useConfig = () => {
  const context = useContext(ConfigContext);
  if (!context) {
    throw new Error('useConfig must be used within a ConfigProvider');
  }
  return context;
};

// Get production API URL from environment variables (set in Vercel dashboard)
// Default to your Render backend URL
const PROD_API_URL = process.env.REACT_APP_API_BASE_URL || 'https://trafficflow.onrender.com';
const PROD_WS_URL = process.env.REACT_APP_WS_BASE_URL || 'wss://trafficflow.onrender.com';

export const ConfigProvider = ({ children }) => {
  const [config, setConfig] = useState({
    API_BASE_URL: 'http://localhost:8000',
    WS_BASE_URL: 'ws://localhost:8000',
    POLL_INTERVAL: 2000,
    STREAM_POLL_INTERVAL: 1000,
    CONFIG_LOADED: false
  });

  const isLocalEnvironment = () => {
    const hostname = window.location.hostname;
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '';
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    // For production (Vercel), use environment variables directly
    if (!isLocalEnvironment()) {
      setConfig({
        API_BASE_URL: PROD_API_URL,
        WS_BASE_URL: PROD_WS_URL,
        POLL_INTERVAL: 2000,
        STREAM_POLL_INTERVAL: 1000,
        CONFIG_LOADED: true
      });
      console.log('☁️ Production environment - API:', PROD_API_URL);
      return;
    }

    // For local development, try to load from backend
    try {
      const configUrl = 'http://localhost:8000/api/config';
      console.log('🔧 Loading config from:', configUrl);
      
      const response = await fetch(configUrl, {
        method: 'GET',
        cache: 'no-cache'
      });
      
      if (response.ok) {
        const data = await response.json();
        setConfig({
          API_BASE_URL: data.api_base_url,
          WS_BASE_URL: data.ws_base_url,
          POLL_INTERVAL: data.poll_intervals?.stats || 2000,
          STREAM_POLL_INTERVAL: data.poll_intervals?.stream_frame || 1000,
          CONFIG_LOADED: true
        });
        console.log('✅ Config loaded from backend:', data);
      } else {
        throw new Error(`Config endpoint returned ${response.status}`);
      }
    } catch (error) {
      console.warn('⚠️ Failed to load config from backend, using defaults:', error.message);
      setConfig(prev => ({
        ...prev,
        API_BASE_URL: 'http://localhost:8000',
        WS_BASE_URL: 'ws://localhost:8000',
        CONFIG_LOADED: true
      }));
      console.log('🏠 Using local development defaults');
    }
  };

  return <ConfigContext.Provider value={config}>{children}</ConfigContext.Provider>;
};
