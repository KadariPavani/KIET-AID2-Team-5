import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';
import { formatTime, formatSpeed } from '../utils/helpers';
import '../styles/analytics.css';

const Dashboard = () => {
  const { getAuthHeaders } = useAuth();
  const { API_BASE_URL, POLL_INTERVAL } = useConfig();
  const [stats, setStats] = useState({
    active_streams: 0,
    total_vehicles: 0,
    total_violations: 0,
    violation_summary: {}
  });
  const [violations, setViolations] = useState([]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [API_BASE_URL, POLL_INTERVAL]);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
        setViolations((data.violations || []).slice(0, 10));
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  return (
    <div className="dashboard-body">
      <Navbar isDashboard={true} />
      

      <div className="dashboard-main">
        <div className="dash-status">
          <h3 className="dash-status-title">Current Status</h3>
          <div className="dash-chips">
            {[
              { icon: 'fa-video', label: 'Streams', value: stats.active_streams || 0 },
              { icon: 'fa-car', label: 'Vehicles', value: stats.total_vehicles || 0 },
              { icon: 'fa-exclamation-triangle', label: 'Total', value: stats.total_violations || 0 },
              { icon: 'fa-traffic-light', label: 'Red Light', value: stats.violation_summary?.red_light || 0 },
              { icon: 'fa-stop', label: 'Stop Line', value: stats.violation_summary?.stop_line || 0 },
              { icon: 'fa-right-left', label: 'Lane', value: stats.violation_summary?.lane_change || 0 },
              { icon: 'fa-car-burst', label: 'Distance', value: stats.violation_summary?.unsafe_distance || 0 },
            ].map((c, i) => (
              <div className="dash-chip" key={i}>
                <i className={`fas ${c.icon}`}></i>
                <span className="dash-chip-val">{c.value}</span>
                <span className="dash-chip-lbl">{c.label}</span>
              </div>
            ))}
          </div>
        </div>

        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-satellite-dish"></i>
              Live Alert Feed
            </h2>
          </div>
          <div className="a-card">
            <div className="a-table-scroll">
              <table className="a-table a-table-simple">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Stream</th>
                    <th>Violation Type</th>
                    <th>Signal State</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {violations.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="a-table-empty">
                        <div>
                          <i className="fas fa-inbox"></i>
                          <p>No violations</p>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    violations.map((v, index) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>Stream {v.stream_id + 1}</td>
                        <td>{(v.violation_type || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                        <td>{v.signal_state || 'N/A'}</td>
                        <td>{formatTime(v.timestamp)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};

export default Dashboard;
