import React from 'react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-traffic-light"></i> TrafficFlow
          </h4>
          <p className="footer-text">
            Intelligent traffic monitoring solutions for smarter, safer cities.
          </p>
        </div>
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-link"></i> Quick Links
          </h4>
          <p className="footer-text"><a href="/dashboard">Dashboard</a></p>
          <p className="footer-text"><a href="/monitoring">Monitoring</a></p>
          <p className="footer-text"><a href="/analytics">Analytics</a></p>
        </div>
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-envelope"></i> Contact Us
          </h4>
          <p className="footer-text">
            <a href="/#contact"><i className="fas fa-paper-plane" style={{marginRight: '6px'}}></i>Send us a message</a>
          </p>
          <p className="footer-text">
            <a href="/dashboard"><i className="fas fa-circle" style={{color: '#00ff88', fontSize: '8px', marginRight: '6px'}}></i>System Online</a>
          </p>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; 2026 TrafficFlow. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
