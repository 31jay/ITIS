import React, { useRef, useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Card, Select, Button, Badge, Statistic, Space, Empty, Tag, Tooltip } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, FullscreenOutlined, CheckOutlined, CameraOutlined, ClockCircleOutlined, RiseOutlined } from '@ant-design/icons';

const { Option } = Select;

interface Detection {
  bbox: number[];
  detection_confidence: number;
  text: string;
  ocr_confidence: number;
  sharpness: number;
  vehicle_bbox?: number[];
}

interface HistoryItem {
  text: string;
  confidence: number;
  timestamp: number;
  image_crop?: string;
}

function App() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [videoList, setVideoList] = useState<string[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [showOverlay, setShowOverlay] = useState(true);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [flash, setFlash] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // Helper to get API URL based on environment
  const getApiUrl = () => {
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    return 'https://smartapi.sanjibkasti.com.np';
  };

  // Helper to get WebSocket URL based on environment
  const getWsUrl = () => {
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'ws://localhost:8000/ws';
    }
    return 'wss://smartapi.sanjibkasti.com.np/ws';
  };

  // Fetch video list on mount
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const response = await axios.get(`${getApiUrl()}/videos`);
        if (response.data.videos && response.data.videos.length > 0) {
          setVideoList(response.data.videos);
          setSelectedVideo(response.data.videos[0]);
        }
      } catch (error) {
        console.error("Error fetching video list:", error);
      }
    };
    fetchVideos();

    // Check streaming status on mount
    const checkStatus = async () => {
      try {
        const response = await axios.get(`${getApiUrl()}/stream/status`);
        if (response.data.is_streaming) {
          setIsStreaming(true);
          if (response.data.source) {
            // Extract filename from full path if needed
            const source = response.data.source;
            const filename = source.includes('/') || source.includes('\\') 
              ? source.split(/[/\\]/).pop() 
              : source;
            setSelectedVideo(filename || source);
          }
        }
      } catch (error) {
        console.error("Error checking stream status:", error);
      }
    };
    checkStatus();
  }, []);

  // Connect to WebSocket
  useEffect(() => {
    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to WebSocket');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'frame') {
        setCurrentImage(`data:image/jpeg;base64,${message.image}`);
        setDetections(message.detections);
        if (message.fps) setFps(message.fps);
        if (message.processing_time_ms) setProcessingTime(message.processing_time_ms);
      } else if (message.type === 'history') {
        setHistory(message.data);
      } else if (message.type === 'new_history') {
        setHistory(prev => [message.data, ...prev].slice(0, 100));
        // Trigger flash animation
        setFlash(true);
        setTimeout(() => setFlash(false), 150);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
    };

    return () => {
      ws.close();
    };
  }, []);

  // Draw on canvas
  useEffect(() => {
    if (!currentImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = currentImage;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      if (showOverlay) {
        detections.forEach(det => {
          // Draw Vehicle Box if available
          if (det.vehicle_bbox) {
            const [vx1, vy1, vx2, vy2] = det.vehicle_bbox;
            ctx.strokeStyle = '#0088FF'; // Blue for vehicle
            ctx.lineWidth = 2;
            ctx.strokeRect(vx1, vy1, vx2 - vx1, vy2 - vy1);
            
            // Vehicle Label
            ctx.fillStyle = '#0088FF';
            ctx.font = '14px Arial';
            const vText = "Vehicle";
            const vTextWidth = ctx.measureText(vText).width;
            ctx.fillRect(vx1, vy1 - 20, vTextWidth + 10, 20);
            ctx.fillStyle = '#FFFFFF';
            ctx.fillText(vText, vx1 + 5, vy1 - 5);
          }

          const [x1, y1, x2, y2] = det.bbox;
          
          // Draw Plate Box
          ctx.strokeStyle = '#00FF00'; // Green for plate
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          // Draw text background
          ctx.fillStyle = '#00FF00';
          const text = `${det.text} (${(det.ocr_confidence * 100).toFixed(1)}%)`;
          const textWidth = ctx.measureText(text).width;
          ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

          // Draw text
          ctx.fillStyle = '#000000';
          ctx.font = '16px Arial';
          ctx.fillText(text, x1 + 5, y1 - 7);
        });
      }
    };
  }, [currentImage, detections, showOverlay]);

  const startStream = async () => {
    try {
      await axios.post(`${getApiUrl()}/stream/start`, { source: selectedVideo });
      setIsStreaming(true);
    } catch (error) {
      console.error("Error starting stream:", error);
    }
  };

  const stopStream = async () => {
    try {
      await axios.post(`${getApiUrl()}/stream/stop`);
      setIsStreaming(false);
    } catch (error) {
      console.error("Error stopping stream:", error);
    }
  };

  const toggleFullScreen = () => {
    if (!videoContainerRef.current) return;

    if (!document.fullscreenElement) {
      videoContainerRef.current.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f5f5f5' }}>
      {/* Header */}
      <div style={{ background: 'white', borderBottom: '1px solid #e8e8e8', padding: '16px 24px' }}>
        <div style={{ maxWidth: '1600px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space size="middle">
            <CameraOutlined style={{ fontSize: '32px', color: '#1890ff' }} />
            <div>
              <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 600 }}>Smart Traffic Management</h1>
              <p style={{ margin: 0, color: '#8c8c8c', fontSize: '14px' }}>AI-Powered ANPR System</p>
            </div>
          </Space>
          <Statistic 
            title="Total Detections" 
            value={history.length} 
            prefix={<CheckOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '1600px', margin: '0 auto', padding: '24px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '24px' }}>
          
          {/* Video Player Section */}
          <div>
            <Card style={{ marginBottom: '16px' }} bodyStyle={{ padding: 0 }}>
              <div 
                ref={videoContainerRef}
                style={{ 
                  position: 'relative', 
                  width: '100%', 
                  paddingTop: '56.25%', 
                  background: '#000',
                  overflow: 'hidden'
                }}
              >
                {currentImage ? (
                  <canvas 
                    ref={canvasRef}
                    style={{ 
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                ) : (
                  <div style={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <div style={{ 
                      width: '64px', 
                      height: '64px', 
                      border: '4px solid #1890ff',
                      borderTopColor: 'transparent',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                      marginBottom: '16px'
                    }} />
                    <p style={{ fontSize: '18px', margin: 0 }}>Waiting for stream...</p>
                  </div>
                )}
                
                {/* Flash Effect */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'white',
                  pointerEvents: 'none',
                  opacity: flash ? 0.4 : 0,
                  transition: 'opacity 150ms',
                  zIndex: 5
                }} />
                
                {/* Stats Overlay */}
                {isStreaming && (
                  <div style={{ position: 'absolute', top: '16px', right: '16px', zIndex: 10 }}>
                    <Space direction="vertical" size="small">
                      <Badge status="processing" text="LIVE" style={{ background: 'rgba(255,255,255,0.95)', padding: '8px 12px', borderRadius: '6px', fontWeight: 600 }} />
                      <Card size="small" style={{ background: 'rgba(255,255,255,0.95)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', fontSize: '12px' }}>
                          <span style={{ color: '#8c8c8c' }}>FPS:</span>
                          <span style={{ fontWeight: 'bold', color: '#52c41a' }}>{fps}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', fontSize: '12px', marginTop: '4px' }}>
                          <span style={{ color: '#8c8c8c' }}>Latency:</span>
                          <span style={{ fontWeight: 'bold', color: '#1890ff' }}>{processingTime}ms</span>
                        </div>
                      </Card>
                    </Space>
                  </div>
                )}

                {/* Video Controls */}
                <div style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
                  padding: '16px',
                  opacity: 0,
                  transition: 'opacity 0.3s',
                  zIndex: 20
                }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '0'}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Space>
                      <Button 
                        type="primary"
                        danger={isStreaming}
                        icon={isStreaming ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                        onClick={isStreaming ? stopStream : startStream}
                        size="large"
                      >
                        {isStreaming ? 'Stop' : 'Start'}
                      </Button>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white', cursor: 'pointer' }}>
                        <input 
                          type="checkbox" 
                          checked={showOverlay} 
                          onChange={(e) => setShowOverlay(e.target.checked)}
                          style={{ width: '16px', height: '16px' }}
                        />
                        <span>Overlays</span>
                      </label>
                    </Space>
                    
                    <Tooltip title="Fullscreen">
                      <Button 
                        type="text" 
                        icon={<FullscreenOutlined style={{ color: 'white', fontSize: '20px' }} />}
                        onClick={toggleFullScreen}
                      />
                    </Tooltip>
                  </div>
                </div>
              </div>
            </Card>

            {/* Video Source Selector */}
            <Card size="small">
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>Video Source</label>
                <Select 
                  style={{ width: '100%' }}
                  size="large"
                  value={selectedVideo}
                  onChange={setSelectedVideo}
                  disabled={isStreaming}
                >
                  <Option value="0">📹 Live Camera (Webcam)</Option>
                  {videoList.map(video => (
                    <Option key={video} value={video}>📁 {video}</Option>
                  ))}
                </Select>
              </div>
            </Card>
          </div>

          {/* Detection History Sidebar */}
          <div>
            <Card 
              title={
                <Space>
                  <RiseOutlined />
                  <span>Detections</span>
                  <Badge count={history.length} showZero style={{ backgroundColor: '#52c41a' }} />
                </Space>
              }
              style={{ height: 'calc(100vh - 120px)' }}
              bodyStyle={{ padding: '12px', height: 'calc(100% - 57px)', overflowY: 'auto' }}
            >
              {history.length === 0 ? (
                <Empty 
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No detections yet"
                  style={{ marginTop: '80px' }}
                />
              ) : (
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  {history.map((item, idx) => (
                    <Card 
                      key={idx} 
                      size="small" 
                      hoverable
                      style={{ 
                        animation: idx === 0 ? 'slideIn 0.3s ease-out' : 'none',
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <div style={{ display: 'flex', gap: '12px' }}>
                        {item.image_crop ? (
                          <div style={{ 
                            width: '80px', 
                            height: '48px', 
                            background: '#f5f5f5', 
                            borderRadius: '4px',
                            overflow: 'hidden',
                            flexShrink: 0,
                            border: '1px solid #d9d9d9'
                          }}>
                            <img 
                              src={`data:image/jpeg;base64,${item.image_crop}`} 
                              alt="Plate" 
                              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            />
                          </div>
                        ) : (
                          <div style={{ 
                            width: '80px', 
                            height: '48px', 
                            background: '#f5f5f5', 
                            borderRadius: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            border: '1px solid #d9d9d9'
                          }}>
                            <CameraOutlined style={{ color: '#bfbfbf' }} />
                          </div>
                        )}
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between', gap: '8px', marginBottom: '4px' }}>
                            <Tooltip title={item.text}>
                              <p style={{ 
                                fontFamily: 'monospace', 
                                fontWeight: 'bold', 
                                fontSize: '16px',
                                margin: 0,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}>
                                {item.text || 'Unknown'}
                              </p>
                            </Tooltip>
                            <Tag color={item.confidence > 0.8 ? 'success' : item.confidence > 0.5 ? 'warning' : 'error'}>
                              {(item.confidence * 100).toFixed(0)}%
                            </Tag>
                          </div>
                          <Space size={4} style={{ fontSize: '12px', color: '#8c8c8c' }}>
                            <ClockCircleOutlined />
                            <span>{new Date(item.timestamp * 1000).toLocaleTimeString()}</span>
                          </Space>
                        </div>
                      </div>
                    </Card>
                  ))}
                </Space>
              )}
            </Card>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        .video-container:hover > div:last-child {
          opacity: 1 !important;
        }
      `}</style>
    </div>
  );
}

export default App;