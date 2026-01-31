import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import AuthModal from '../components/AuthModal';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';

const Landing = () => {
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authTab, setAuthTab] = useState('login');
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const { API_BASE_URL } = useConfig();
  
  // Contact form state
  const [contactForm, setContactForm] = useState({ name: '', email: '', subject: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  const handleContactChange = (e) => {
    const { name, value } = e.target;
    setContactForm(prev => ({ ...prev, [name]: value }));
  };

  const handleContactSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/contact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(contactForm)
      });
      if (response.ok) {
        setSubmitStatus('success');
        setContactForm({ name: '', email: '', subject: '', message: '' });
      } else {
        setSubmitStatus('error');
      }
    } catch (error) {
      console.error('Error submitting contact form:', error);
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/monitoring');
    }
  }, [isAuthenticated, navigate]);

  const openSignup = () => {
    setAuthTab('signup');
    setShowAuthModal(true);
  };

  const openLogin = () => {
    setAuthTab('login');
    setShowAuthModal(true);
  };

  return (
    <>
      <Navbar isDashboard={false} />
      
      <main className="main-content landing-page">
        <section id="home" className="landing-container">
          <div className="landing-hero">
            <div className="hero-logo">
              <div className="logo-large">
                <div className="logo-icon">TrafficFlow</div>
                <div className="signal-pulse"></div>
              </div>
            </div>

            <h1 className="hero-title">AI Traffic Monitoring System</h1>
            <p className="hero-subtitle">
              Real-time vehicle detection, speed monitoring, and violation tracking to make roads
              safer and smarter using advanced computer vision and deep learning.
            </p>

            <div className="hero-actions">
              <button className="btn btn-large btn-primary" onClick={openSignup}>
                <i className="fas fa-user-plus"></i> Get Started – Sign Up
              </button>
              <button className="btn btn-secondary btn-large" onClick={openLogin}>
                <i className="fas fa-right-to-bracket"></i> Already have an account? Login
              </button>
            </div>

            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-number">4+</div>
                <div className="stat-text">Concurrent Streams</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">6</div>
                <div className="stat-text">Violation Types</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">24/7</div>
                <div className="stat-text">Continuous Monitoring</div>
              </div>
            </div>
          </div>
        </section>

        <section id="about" className="grid-section">
          <div className="section-title">
            <i className="fas fa-circle-info"></i>
            About the Project
          </div>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Traffic Monitoring System (TMS)</div>
            </div>
            <p>
              The Traffic Monitoring System is an end-to-end solution that uses YOLOv8-based
              object detection and custom computer vision logic to automatically monitor
              road traffic in real time. It tracks vehicles across multiple streams,
              estimates their speed, detects violations, and stores structured records in
              a database for analysis and reporting.
            </p>
            <p>
              This project is designed for smart cities, traffic departments, and
              academic research, demonstrating how AI can be integrated with live streams
              to improve safety, enforcement, and traffic flow management.
            </p>
          </div>
        </section>

        <section id="project" className="grid-section">
          <div className="section-title">
            <i className="fas fa-project-diagram"></i>
            Project Overview & Features
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <i className="fas fa-brain stat-icon"></i>
              <div className="stat-label">Core Technology</div>
              <div className="stat-unit">
                YOLOv8 object detection, OpenCV-based tracking, FastAPI backend, MongoDB storage,
                and a modern dashboard frontend.
              </div>
            </div>
            <div className="stat-card">
              <i className="fas fa-shield-alt stat-icon"></i>
              <div className="stat-label">Violation Detection</div>
              <div className="stat-unit">
                Speeding, red light jump, unsafe following distance, lane misuse, and
                stop-line violations with image evidence.
              </div>
            </div>
            <div className="stat-card">
              <i className="fas fa-gauge-high stat-icon"></i>
              <div className="stat-label">Live Analytics</div>
              <div className="stat-unit">
                Real-time statistics on total vehicles, active streams, and violations
                with a rich visualization dashboard.
              </div>
            </div>
            <div className="stat-card">
              <i className="fas fa-users stat-icon"></i>
              <div className="stat-label">User Access</div>
              <div className="stat-unit">
                Secure signup and login, JWT-based authentication, and protected
                dashboard access after successful login.
              </div>
            </div>
          </div>
        </section>

        <section id="contact" className="grid-section">
          <div className="section-title">
            <i className="fas fa-envelope"></i>
            Contact Us
          </div>
          <div className="card contact-card">
            {submitStatus === 'success' ? (
              <div className="success-message">
                <div className="success-icon">
                  <i className="fas fa-check-circle"></i>
                </div>
                <h3>Thank You!</h3>
                <p>Your message has been successfully submitted. We'll get back to you soon.</p>
                <button className="btn btn-primary" onClick={() => setSubmitStatus(null)}>
                  Send Another Message
                </button>
              </div>
            ) : (
              <form onSubmit={handleContactSubmit} className="landing-contact-form">
                {submitStatus === 'error' && (
                  <div className="error-alert">
                    <i className="fas fa-exclamation-circle"></i>
                    Failed to submit. Please try again.
                  </div>
                )}
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="name"><i className="fas fa-user"></i> Name</label>
                    <input type="text" id="name" name="name" value={contactForm.name} onChange={handleContactChange} placeholder="Your name" required />
                  </div>
                  <div className="form-group">
                    <label htmlFor="email"><i className="fas fa-envelope"></i> Email</label>
                    <input type="email" id="email" name="email" value={contactForm.email} onChange={handleContactChange} placeholder="Your email" required />
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor="subject"><i className="fas fa-tag"></i> Subject</label>
                  <input type="text" id="subject" name="subject" value={contactForm.subject} onChange={handleContactChange} placeholder="What is this regarding?" required />
                </div>
                <div className="form-group">
                  <label htmlFor="message"><i className="fas fa-comment-alt"></i> Message</label>
                  <textarea id="message" name="message" value={contactForm.message} onChange={handleContactChange} placeholder="Write your message here..." rows="4" required></textarea>
                </div>
                <button type="submit" className="btn btn-primary btn-submit" disabled={isSubmitting}>
                  {isSubmitting ? (<><i className="fas fa-spinner fa-spin"></i> Sending...</>) : (<><i className="fas fa-paper-plane"></i> Send Message</>)}
                </button>
              </form>
            )}
          </div>
        </section>
      </main>

      <Footer />

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialTab={authTab}
      />
    </>
  );
};

export default Landing;
