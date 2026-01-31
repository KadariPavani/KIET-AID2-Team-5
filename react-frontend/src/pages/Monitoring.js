import React, { useState, useEffect, useRef, useCallback } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';
import { showToast } from '../utils/helpers';

const StreamCard = ({ streamId, isMain, onSwap, API_BASE_URL, getAuthHeaders, STREAM_POLL_INTERVAL }) => {
  const [status, setStatus] = useState('INACTIVE');
  const [streamUrl, setStreamUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const imgRef = useRef(null);
  const placeholderRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    reconnectIfActive();
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [streamId]);

  const reconnectIfActive = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stream-status/${streamId}`, {
        headers: getAuthHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        if (data.processing) {
          setStatus('ACTIVE');
          setIsStreaming(true);
          if (data.stream_url) {
            setStreamUrl(data.stream_url);
          }
          startPolling();
        }
      }
    } catch (error) {
      console.error(`Error checking stream ${streamId} status:`, error);
    }
  };

  const startStream = async () => {
    if (!streamUrl.trim()) {
      showToast('Please enter a stream URL', 'error');
      return;
    }

    const isYouTube = streamUrl.includes('youtube.com') || streamUrl.includes('youtu.be');
    
    if (isYouTube) {
      showToast(`Processing YouTube URL for stream ${streamId + 1}...`, 'info');
    } else {
      showToast(`Starting stream ${streamId + 1}...`, 'info');
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/start-stream/${streamId}?stream_url=${encodeURIComponent(streamUrl)}`,
        {
          method: 'POST',
          headers: getAuthHeaders()
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || `Server returned ${response.status}`);
      }

      setStatus('ACTIVE');
      setIsStreaming(true);
      showToast(`Stream ${streamId + 1} started successfully!`, 'success');
      
      if (placeholderRef.current) {
        placeholderRef.current.style.display = 'flex';
        placeholderRef.current.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Processing video stream...</p>';
      }

      setTimeout(startPolling, isYouTube ? 5000 : 500);
    } catch (error) {
      showToast(`Failed to start stream: ${error.message}`, 'error');
    }
  };

  const stopStream = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stop-stream/${streamId}`, {
        method: 'POST',
        headers: getAuthHeaders()
      });

      if (response.ok) {
        setStatus('INACTIVE');
        setIsStreaming(false);
        showToast(`Stream ${streamId + 1} stopped`, 'success');

        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }

        if (imgRef.current) imgRef.current.style.display = 'none';
        if (placeholderRef.current) {
          placeholderRef.current.style.display = 'flex';
          placeholderRef.current.innerHTML = '<i class="fas fa-video"></i><p>Stream Inactive</p>';
        }
      }
    } catch (error) {
      showToast(`Failed to stop stream: ${error.message}`, 'error');
    }
  };

  const startPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    let consecutiveErrors = 0;
    const maxErrors = 5;

    pollIntervalRef.current = setInterval(async () => {
      try {
        const timestamp = new Date().getTime();
        const frameUrl = `${API_BASE_URL}/stream/${streamId}/frame?t=${timestamp}`;

        const response = await fetch(frameUrl, {
          headers: getAuthHeaders(),
          cache: 'no-store',
          signal: AbortSignal.timeout(10000)
        });

        if (response.ok && response.headers.get('content-type')?.includes('image')) {
          consecutiveErrors = 0;
          const blob = await response.blob();
          const imageUrl = URL.createObjectURL(blob);

          if (imgRef.current) {
            imgRef.current.onload = () => URL.revokeObjectURL(imageUrl);
            imgRef.current.src = imageUrl;
            imgRef.current.style.display = 'block';
          }
          if (placeholderRef.current) {
            placeholderRef.current.style.display = 'none';
          }
        } else {
          consecutiveErrors++;
          if (consecutiveErrors >= maxErrors) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
            showToast(`Stream ${streamId + 1} connection lost`, 'error');
          }
        }
      } catch (error) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxErrors) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      }
    }, STREAM_POLL_INTERVAL);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      showToast('Please select a video file', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const headers = {};
      const authHeaders = getAuthHeaders();
      if (authHeaders.Authorization) {
        headers.Authorization = authHeaders.Authorization;
      }

      const response = await fetch(`${API_BASE_URL}/api/upload-video/${streamId}`, {
        method: 'POST',
        headers,
        body: formData
      });

      if (response.ok) {
        showToast(`Video processing started for Stream ${streamId + 1}`, 'success');
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        setStatus('ACTIVE');
        setIsStreaming(true);
        startPolling();
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      showToast(`Upload failed: ${error.message}`, 'error');
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleCardClick = (e) => {
    // Only trigger swap if not clicking on input, button, or label
    if (
      e.target.tagName !== 'INPUT' &&
      e.target.tagName !== 'BUTTON' &&
      !e.target.closest('button') &&
      !e.target.closest('label') &&
      !isMain
    ) {
      onSwap(streamId);
    }
  };

  return (
    <div
      className={`stream-card-wrapper ${isMain ? 'big main-stream' : 'small'}`}
      data-stream={streamId}
      onClick={handleCardClick}
      style={{ cursor: isMain ? 'default' : 'pointer' }}
    >
      <div className="stream-card-content">
        <div className="stream-header">
          <h3>Stream {streamId + 1}{isMain ? ' (Main)' : ''}</h3>
          <span className={`status-badge status-${status.toLowerCase()}`}>
            {status}
          </span>
        </div>

        <div className="stream-video">
          <div
            ref={placeholderRef}
            className="video-placeholder"
            style={{ display: 'flex' }}
          >
            <i className="fas fa-video"></i>
            <p>{isMain ? 'Main Stream Inactive' : 'Stream Inactive'}</p>
          </div>
          <img
            ref={imgRef}
            className="video-feed"
            style={{ display: 'none' }}
            alt={`Stream ${streamId}`}
          />
        </div>

        <div className="stream-controls">
          <input
            type="text"
            className="stream-url"
            placeholder="RTSP/HTTP/YouTube URL"
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            onClick={(e) => e.stopPropagation()}
          />

          <div className="upload-section">
            <label htmlFor={`videoFile${streamId}`} className="upload-label" onClick={(e) => e.stopPropagation()}>
              <div className="upload-box">
                <i className="fas fa-film"></i>
                <span className="upload-title">Upload Video</span>
                <p className="upload-subtitle">Click or drag to upload</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                id={`videoFile${streamId}`}
                className="file-input"
                accept="video/*"
                onChange={handleFileChange}
              />
            </label>
            {selectedFile && (
              <div className="selected-file" style={{ display: 'flex' }}>
                <i className="fas fa-check-circle"></i>
                <span>{selectedFile.name}</span>
                <button
                  type="button"
                  className="file-clear-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearFile();
                  }}
                >
                  <i className="fas fa-times"></i>
                </button>
              </div>
            )}
          </div>

          <div className="stream-buttons">
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.stopPropagation();
                startStream();
              }}
            >
              <i className="fas fa-play"></i> Start
            </button>
            <button
              className="btn btn-danger"
              onClick={(e) => {
                e.stopPropagation();
                stopStream();
              }}
            >
              <i className="fas fa-stop"></i> Stop
            </button>
            <button
              className="btn btn-accent"
              onClick={(e) => {
                e.stopPropagation();
                handleFileUpload();
              }}
            >
              <i className="fas fa-upload"></i> Upload
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// const Monitoring = () => {
//   const { getAuthHeaders } = useAuth();
//   const { API_BASE_URL, STREAM_POLL_INTERVAL } = useConfig();
//   const [mainStreamIndex, setMainStreamIndex] = useState(2);

//   const handleSwapToMain = (streamIndex) => {
//     if (streamIndex !== mainStreamIndex) {
//       setMainStreamIndex(streamIndex);
//       showToast(`Stream ${streamIndex + 1} moved to main view`, 'success');
//     }
//   };

//   const smallStreams = [0, 1, 3].filter(i => i !== mainStreamIndex);
//   if (!smallStreams.includes(mainStreamIndex) && mainStreamIndex !== 2) {
//     smallStreams.push(mainStreamIndex);
//   }

//   return (
//     <div className="dashboard-body">
//       <Navbar isDashboard={true} />

//       <div className="dashboard-main">
//         <section className="content-section">
//           <div className="section-header">
//             <h2 className="section-title">
//               <i className="fas fa-satellite"></i>
//               Stream Configuration & Live Feed
//             </h2>
//           </div>

//           <div className="monitoring-layout">
//             <div className="small-streams-panel">
//               {[0, 1, 3].map(streamId => (
//                 streamId !== mainStreamIndex && (
//                   <StreamCard
//                     key={streamId}
//                     streamId={streamId}
//                     isMain={false}
//                     onSwap={handleSwapToMain}
//                     API_BASE_URL={API_BASE_URL}
//                     getAuthHeaders={getAuthHeaders}
//                     STREAM_POLL_INTERVAL={STREAM_POLL_INTERVAL}
//                   />
//                 )
//               ))}
//             </div>

//             <div className="big-stream-panel">
//               <StreamCard
//                 key={`main-${mainStreamIndex}`}
//                 streamId={mainStreamIndex}
//                 isMain={true}
//                 onSwap={() => {}}
//                 API_BASE_URL={API_BASE_URL}
//                 getAuthHeaders={getAuthHeaders}
//                 STREAM_POLL_INTERVAL={STREAM_POLL_INTERVAL}
//               />
//             </div>
//           </div>
//         </section>
//       </div>

//       <Footer />
//     </div>
//   );
// };

const Monitoring = () => {
  const { getAuthHeaders } = useAuth();
  const { API_BASE_URL, STREAM_POLL_INTERVAL } = useConfig();

  /**
   * SOURCE OF TRUTH
   * Index 2 is ALWAYS the main stream (matches script.js)
   */
  const MAIN_INDEX = 2;

  const [streams, setStreams] = useState([
    { id: 0 },
    { id: 1 },
    { id: 2 }, // MAIN by default
    { id: 3 }
  ]);

  /**
   * Swap clicked stream with main stream
   * React-safe equivalent of DOM swap in script.js
   */
  const handleSwapToMain = (clickedStreamId) => {
    const clickedIndex = streams.findIndex(s => s.id === clickedStreamId);
    if (clickedIndex === -1 || clickedIndex === MAIN_INDEX) return;

    setStreams(prev => {
      const updated = [...prev];

      // Swap stream objects
      const temp = updated[MAIN_INDEX];
      updated[MAIN_INDEX] = updated[clickedIndex];
      updated[clickedIndex] = temp;

      return updated;
    });

    showToast(`Stream ${clickedStreamId + 1} moved to main view`, 'success');
  };

  // WebSocket for real-time violation notifications
  useEffect(() => {
    const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss' : 'ws';
    const wsHost = new URL(API_BASE_URL).host;
    const wsUrl = `${wsProtocol}://${wsHost}/ws`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected for violations');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'violation') {
          const { data } = message;
          const violationType = data.violation_type.replace('_', ' ').toUpperCase();
          showToast(`Violation detected on Stream ${data.stream_id + 1}: ${violationType}`, 'error');
          
          // Play notification alert sound using Web Audio API
          try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            // Notification-style beep pattern
            oscillator.frequency.setValueAtTime(880, audioContext.currentTime); // A5 note
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
            
            // Second beep
            setTimeout(() => {
              const osc2 = audioContext.createOscillator();
              const gain2 = audioContext.createGain();
              osc2.connect(gain2);
              gain2.connect(audioContext.destination);
              osc2.frequency.setValueAtTime(1100, audioContext.currentTime); // Higher pitch
              osc2.type = 'sine';
              gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
              gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
              osc2.start(audioContext.currentTime);
              osc2.stop(audioContext.currentTime + 0.3);
            }, 150);
          } catch (audioError) {
            console.error('Error playing notification sound:', audioError);
          }
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [API_BASE_URL]);

  return (
    <div className="dashboard-body">
      <Navbar isDashboard={true} />

      <div className="dashboard-main">
        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-satellite"></i>
              Stream Configuration & Live Feed
            </h2>
          </div>

          <div className="monitoring-layout">
            {/* SMALL STREAMS */}
            <div className="small-streams-panel">
              {streams.map((stream, index) => {
                if (index === MAIN_INDEX) return null;

                return (
                  <StreamCard
                    key={stream.id}                // ✅ stable identity
                    streamId={stream.id}
                    isMain={false}
                    onSwap={handleSwapToMain}
                    API_BASE_URL={API_BASE_URL}
                    getAuthHeaders={getAuthHeaders}
                    STREAM_POLL_INTERVAL={STREAM_POLL_INTERVAL}
                  />
                );
              })}
            </div>

            {/* MAIN STREAM */}
            <div className="big-stream-panel">
              <StreamCard
                key={streams[MAIN_INDEX].id}
                streamId={streams[MAIN_INDEX].id}
                isMain={true}
                onSwap={() => {}}
                API_BASE_URL={API_BASE_URL}
                getAuthHeaders={getAuthHeaders}
                STREAM_POLL_INTERVAL={STREAM_POLL_INTERVAL}
              />
            </div>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};


export default Monitoring;