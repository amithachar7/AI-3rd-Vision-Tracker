import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  Camera,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Eye,
  FileVideo,
  Gauge,
  LoaderCircle,
  Menu,
  Play,
  Radio,
  Scan,
  Server,
  ShieldCheck,
  Upload,
  Users,
  Wifi,
  XCircle,
  Zap,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [device, setDevice] = useState("Unavailable");
  const [activePage, setActivePage] = useState("dashboard");

  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [videoResult, setVideoResult] = useState(null);
  const [videoError, setVideoError] = useState("");

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [cameraImage, setCameraImage] = useState("");
  const [cameraDetections, setCameraDetections] = useState([]);
  const [cameraFps, setCameraFps] = useState(0);
  const [cameraTrackingIds, setCameraTrackingIds] = useState(0);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const cameraActiveRef = useRef(false);
  const cameraBusyRef = useRef(false);
  const frameTimesRef = useRef([]);
  const cameraTimerRef = useRef(null);

  const checkBackend = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/health`, {
        signal: AbortSignal.timeout(4000),
      });

      if (!response.ok) throw new Error("Backend unavailable");

      const data = await response.json();
      setBackendOnline(true);
      setDevice(data.device?.toUpperCase() || "CPU");
    } catch {
      setBackendOnline(false);
      setDevice("Unavailable");
    }
  }, []);

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 8000);
    return () => clearInterval(interval);
  }, [checkBackend]);

  const navigateTo = (page, elementId) => {
    setActivePage(page);
    document.getElementById(elementId)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const chooseFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setVideoResult(null);
    setVideoError("");
  };

  const processVideo = async (event) => {
    event.preventDefault();
    if (!selectedFile || processing) return;

    setProcessing(true);
    setVideoError("");
    setVideoResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const uploadResponse = await fetch(`${API}/api/process-video`, {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadResponse.json();

      if (!uploadResponse.ok) {
        throw new Error(uploadData.detail || "Video upload failed.");
      }
      if (!uploadData.task_id) {
        throw new Error("Backend did not return a processing task ID.");
      }

      // The backend processes videos in the background. Poll its task endpoint.
      let finished = false;
      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 800));

        const progressResponse = await fetch(
          `${API}/api/process-video/${uploadData.task_id}`
        );
        const progressData = await progressResponse.json();

        if (!progressResponse.ok) {
          throw new Error(progressData.detail || "Could not read processing status.");
        }

        if (progressData.status === "failed") {
          throw new Error(progressData.error || "Video processing failed.");
        }

        if (progressData.status === "completed") {
          if (!progressData.result) {
            throw new Error("Processing finished but no result was returned.");
          }
          setVideoResult(progressData.result);
          finished = true;
        }
      }
    } catch (error) {
      setVideoError(error.message || "Could not process the video. Check the backend.");
    } finally {
      setProcessing(false);
      checkBackend();
    }
  };

  const stopCamera = useCallback(() => {
    cameraActiveRef.current = false;
    setCameraActive(false);

    if (cameraTimerRef.current) {
      clearTimeout(cameraTimerRef.current);
      cameraTimerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraImage("");
    setCameraDetections([]);
    setCameraFps(0);
    setCameraTrackingIds(0);
    cameraBusyRef.current = false;
  }, []);

  const sendCameraFrame = useCallback(async () => {
    if (!cameraActiveRef.current || cameraBusyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      if (cameraActiveRef.current) {
        cameraTimerRef.current = setTimeout(sendCameraFrame, 180);
      }
      return;
    }

    cameraBusyRef.current = true;

    try {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) return;

      // Downscale camera frames before upload to reduce GPU/API queue latency.
      const maxWidth = 800;
      const scale = Math.min(1, maxWidth / width);
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);

      const context = canvas.getContext("2d", { alpha: false });
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.76)
      );
      if (!blob || !cameraActiveRef.current) return;

      const formData = new FormData();
      formData.append("file", blob, "webcam-frame.jpg");

      // Correct endpoint: this route uses persistent ByteTrack tracking IDs.
      const response = await fetch(`${API}/api/webcam/detect`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Camera detection request failed.");
      }
      if (!cameraActiveRef.current) return;

      if (data.image) {
        setCameraImage(`data:image/jpeg;base64,${data.image}`);
      }
      setCameraDetections(data.detections || []);
      setCameraTrackingIds(Number(data.unique_tracking_ids || 0));

      const now = performance.now();
      frameTimesRef.current.push(now);
      frameTimesRef.current = frameTimesRef.current.filter(
        (time) => now - time < 4000
      );
      const samples = frameTimesRef.current.length;
      if (samples > 1) {
        const elapsedSeconds = (now - frameTimesRef.current[0]) / 1000;
        setCameraFps(elapsedSeconds > 0 ? Math.round((samples - 1) / elapsedSeconds) : 0);
      }
    } catch (error) {
      if (cameraActiveRef.current) {
        setCameraError(error.message || "Webcam detection failed.");
        stopCamera();
      }
    } finally {
      cameraBusyRef.current = false;
      if (cameraActiveRef.current) {
        // Run one request at a time; no overlapping frames or flickering loader.
        cameraTimerRef.current = setTimeout(sendCameraFrame, 180);
      }
    }
  }, [stopCamera]);

  const startCamera = async () => {
    if (cameraActiveRef.current) return;

    setCameraError("");
    setCameraImage("");
    setCameraDetections([]);
    setCameraFps(0);
    setCameraTrackingIds(0);
    frameTimesRef.current = [];

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires localhost or HTTPS.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 800 },
          height: { ideal: 600 },
          frameRate: { ideal: 24, max: 30 },
        },
        audio: false,
      });

      streamRef.current = stream;
      cameraActiveRef.current = true;
      setCameraActive(true);

      const video = videoRef.current;
      if (!video) throw new Error("Camera preview element is unavailable.");

      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;

      // Wait for metadata before starting playback. Do not repeatedly call load().
      if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise((resolve, reject) => {
          const onReady = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error("Could not initialize the webcam preview."));
          };
          const cleanup = () => {
            video.removeEventListener("loadedmetadata", onReady);
            video.removeEventListener("error", onError);
          };
          video.addEventListener("loadedmetadata", onReady, { once: true });
          video.addEventListener("error", onError, { once: true });
        });
      }

      if (!cameraActiveRef.current || video.srcObject !== stream) return;

      try {
        await video.play();
      } catch (playError) {
        // Stop/restart can cancel a pending play promise; that is not a camera failure.
        if (playError.name !== "AbortError" && cameraActiveRef.current) {
          throw playError;
        }
      }

      if (cameraActiveRef.current) {
        cameraTimerRef.current = setTimeout(sendCameraFrame, 250);
      }
    } catch (error) {
      const message = error.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access in your browser."
        : error.message || "Unable to start camera.";
      stopCamera();
      setCameraError(message);
    }
  };

  useEffect(() => {
    return () => {
      cameraActiveRef.current = false;
      if (cameraTimerRef.current) clearTimeout(cameraTimerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const categories = videoResult?.object_categories || {};
  const categoryEntries = Object.entries(categories);
  const totalCategoryDetections = categoryEntries.reduce(
    (sum, [, count]) => sum + Number(count || 0),
    0
  );

  const downloadUrl = videoResult?.download_url
    ? videoResult.download_url.startsWith("http")
      ? videoResult.download_url
      : `${API}${videoResult.download_url}`
    : "";

  const formatSize = (bytes) => {
    if (!bytes) return "0 MB";
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Scan size={24} />
          </div>
          <div>
            <div className="brand-name">AI 3rd</div>
            <div className="brand-subtitle">VISION TRACKER</div>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="side-nav">
          <button
            className={activePage === "dashboard" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("dashboard", "dashboard")}
          >
            <Activity size={18} />
            Dashboard
          </button>

          <button
            className={activePage === "studio" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("studio", "studio")}
          >
            <Scan size={18} />
            Detection Studio
          </button>

          <button
            className={activePage === "camera" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("camera", "camera")}
          >
            <Camera size={18} />
            Live Camera
          </button>

          <button
            className={activePage === "system" ? "nav-item active" : "nav-item"}
            onClick={() => navigateTo("system", "system")}
          >
            <Cpu size={18} />
            System Status
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-security">
            <ShieldCheck size={19} />
            <div>
              <strong>AI Processing</strong>
              <span>Local GPU acceleration</span>
            </div>
          </div>
          <div className="sidebar-version">VISION TRACKER Â· 1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">WORKSPACE / OVERVIEW</div>
            <h1>Vision Dashboard</h1>
          </div>

          <div className="topbar-actions">
            <div className={backendOnline ? "status-pill online" : "status-pill offline"}>
              <span className="status-dot" />
              {backendOnline ? "BACKEND ONLINE" : "BACKEND OFFLINE"}
            </div>
            <div className="topbar-chip" title="Processing device">
              <Cpu size={20} />
            </div>
          </div>
        </header>

        <section className="hero-panel" id="dashboard">
          <div className="hero-copy">
            <div className="eyebrow">
              <Radio size={14} />
              INTELLIGENT VIDEO ANALYTICS
            </div>

            <h2>
              See Smarter.
              <br />
              <span>Track Faster.</span>
            </h2>

            <p>
              Detect objects, track movement and analyze video footage using
              real-time computer vision.
            </p>

            <button
              className="primary-button"
              onClick={() => navigateTo("studio", "studio")}
            >
              <Play size={16} fill="currentColor" />
              Open Detection Studio
            </button>
          </div>

          <div className="welcome-graphic">
            <div className="scanner">
              <div className="scanner-ring ring-one" />
              <div className="scanner-ring ring-two" />
              <div className="scanner-ring ring-three" />
              <div className="scanner-sweep" />

              <div className="scanner-crosshair">
                <i className="corner top-left" />
                <i className="corner top-right" />
                <i className="corner bottom-left" />
                <i className="corner bottom-right" />
              </div>

              <div className="scanner-center" />
              <i className="target-marker marker-one" />
              <i className="target-marker marker-two" />
            </div>
            <div className="graphic-caption">YOLO11 OBJECT DETECTION</div>
          </div>
        </section>

        <section className="stats-grid" id="system">
          <div className="stat-card">
            <div className="stat-icon cyan"><Server size={20} /></div>
            <div className="stat-label">Backend Status</div>
            <div className="stat-value">
              {backendOnline ? "Connected" : "Offline"}
            </div>
            <div className="stat-foot">FastAPI server</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon purple"><Cpu size={20} /></div>
            <div className="stat-label">Processing Device</div>
            <div className="stat-value">{device}</div>
            <div className="stat-foot">PyTorch acceleration</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon blue"><Eye size={20} /></div>
            <div className="stat-label">Detection Model</div>
            <div className="stat-value">YOLO11s</div>
            <div className="stat-foot">Object detection</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon green"><Gauge size={20} /></div>
            <div className="stat-label">Tracking Engine</div>
            <div className="stat-value">ByteTrack</div>
            <div className="stat-foot">Video object ID tracking</div>
          </div>
        </section>

        <section className="studio-section" id="studio">
          <div className="section-heading">
            <div>
              <div className="section-eyebrow">GET STARTED</div>
              <h2>Detection Studio</h2>
              <p>Process a video and export a detailed detection report.</p>
            </div>
            <div className="engine-pill">
              <Activity size={14} />
              YOLO11s ENGINE
            </div>
          </div>

          <div className="work-grid">
            <div className="work-card video-card">
              <div className="card-heading">
                <div className="card-icon cyan"><FileVideo size={21} /></div>
                <div>
                  <h3>Video Analysis</h3>
                  <p>Upload a video for object detection and tracking</p>
                </div>
              </div>

              <form onSubmit={processVideo}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv"
                  hidden
                  onChange={chooseFile}
                />

                <button
                  type="button"
                  className="upload-zone"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={processing}
                >
                  <div className="upload-symbol"><Upload size={25} /></div>
                  <strong>
                    {selectedFile ? selectedFile.name : "Choose or drop a video"}
                  </strong>
                  <span>
                    {selectedFile
                      ? formatSize(selectedFile.size)
                      : "MP4 Â· AVI Â· MOV Â· MKV"}
                  </span>
                  <small>Click to browse files</small>
                </button>

                {selectedFile && (
                  <div className="selected-file">
                    <FileVideo size={16} />
                    <span>{selectedFile.name}</span>
                    <button
                      type="button"
                      aria-label="Remove selected video"
                      onClick={() => {
                        setSelectedFile(null);
                        setVideoResult(null);
                        setVideoError("");
                        if (fileInputRef.current) fileInputRef.current.value = "";
                      }}
                    >
                      <XCircle size={17} />
                    </button>
                  </div>
                )}

                <button
                  type="submit"
                  className="primary-button analyze-button"
                  disabled={!selectedFile || processing || !backendOnline}
                >
                  {processing ? (
                    <>
                      <LoaderCircle className="spin" size={17} />
                      Processing video...
                    </>
                  ) : (
                    <>
                      <Play size={16} fill="currentColor" />
                      Start Video Analysis
                    </>
                  )}
                </button>
              </form>

              {processing && (
                <div className="processing-message">
                  <LoaderCircle className="spin" size={18} />
                  <div>
                    <strong>AI analysis in progress</strong>
                    <span>
                      Processing video frames on {device}. Please keep this
                      page open.
                    </span>
                  </div>
                </div>
              )}

              {videoError && (
                <div className="error-message">
                  <XCircle size={18} />
                  {videoError}
                </div>
              )}

              {videoResult && (
                <div className="result-panel">
                  <div className="result-title">
                    <CheckCircle2 size={19} />
                    Processing completed
                  </div>

                  <div className="result-metrics">
                    <div>
                      <strong>{videoResult.frames_processed ?? 0}</strong>
                      <span>Frames processed</span>
                    </div>
                    <div>
                      <strong>{videoResult.unique_tracking_ids ?? 0}</strong>
                      <span>Unique tracking IDs</span>
                    </div>
                    <div>
                      <strong>{videoResult.total_frames ?? 0}</strong>
                      <span>Total frames</span>
                    </div>
                  </div>

                  <div className="category-heading">
                    Detection categories
                    <span>{categoryEntries.length} categories</span>
                  </div>

                  <div className="category-list">
                    {categoryEntries.map(([name, count]) => (
                      <div className="category-row" key={name}>
                        <span>{name}</span>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>

                  <div className="result-total">
                    <span>Total detection occurrences</span>
                    <strong>{totalCategoryDetections}</strong>
                  </div>

                  {videoResult.processing_seconds != null && (
                    <p className="processing-time">
                      Processing time: {Number(videoResult.processing_seconds).toFixed(2)} seconds
                    </p>
                  )}

                  {downloadUrl && (
                    <a
                      className="download-button"
                      href={downloadUrl}
                      download
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Download size={17} />
                      Download Processed Video
                    </a>
                  )}
                </div>
              )}
            </div>

            <div className="work-card camera-card" id="camera">
              <div className="card-heading">
                <div className="card-icon purple"><Camera size={21} /></div>
                <div>
                  <h3>Live Camera Tracking</h3>
                  <p>Real-time object detection from your webcam</p>
                </div>
              </div>

              <div className="camera-screen">
                <video
                  ref={videoRef}
                  className="camera-source"
                  muted
                  playsInline
                />

                {cameraActive && cameraImage && (
                  <img
                    className="camera-overlay"
                    src={cameraImage}
                    alt="AI annotated webcam detections"
                  />
                )}

                {!cameraActive && (
                  <div className="camera-placeholder">
                    <div className="camera-placeholder-icon">
                      <Camera size={38} />
                    </div>
                    <strong>Camera not started</strong>
                    <span>Start webcam to see detected objects.</span>
                  </div>
                )}

                {cameraActive && (
                  <div className="camera-live-label">
                    <span />
                    LIVE
                  </div>
                )}
              </div>

              <div className="camera-controls">
                {!cameraActive ? (
                  <button
                    className="primary-button camera-start"
                    onClick={startCamera}
                    disabled={!backendOnline}
                  >
                    <Camera size={17} />
                    Start Webcam
                  </button>
                ) : (
                  <button
                    className="stop-button"
                    onClick={stopCamera}
                  >
                    <XCircle size={17} />
                    Stop Webcam
                  </button>
                )}
              </div>

              {cameraError && (
                <div className="error-message">
                  <XCircle size={17} />
                  {cameraError}
                </div>
              )}

              <div className="camera-metrics">
                <div>
                  <Zap size={16} />
                  <strong>{cameraActive ? cameraFps : 0}</strong>
                  <span>FPS</span>
                </div>
                <div>
                  <Eye size={16} />
                  <strong>{cameraDetections.length}</strong>
                  <span>Objects</span>
                </div>
                <div>
                  <Users size={16} />
                  <strong>{cameraTrackingIds}</strong>
                  <span>Tracking IDs</span>
                </div>
              </div>

              <div className="live-analytics">
                <div className="analytics-heading">
                  <strong>Live detection results</strong>
                  <span>{cameraDetections.length} detections</span>
                </div>

                <div className="camera-waiting" aria-live="polite">
                  <span className="status-dot" />
                  {cameraActive ? "Live analysis running · results refresh automatically" : "Analysis is idle"}
                </div>

                {cameraDetections.length > 0 ? (
                  <div className="live-detection-list">
                    {cameraDetections.map((item, index) => (
                      <div className="live-detection" key={`${item.class_name}-${index}`}>
                        <span>{item.class_name}</span>
                        <strong>
                          {Math.round((item.confidence || 0) * 100)}%
                        </strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-detections">
                    {cameraActive
                      ? "Waiting for camera detections..."
                      : "Start the webcam to view live detection results."}
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="system-section">
          <div className="system-heading">
            <div>
              <div className="section-eyebrow">SYSTEM INFORMATION</div>
              <h2>System Status</h2>
            </div>
            <button className="refresh-button" onClick={checkBackend}>
              <Wifi size={16} />
              Refresh status
            </button>
          </div>

          <div className="system-panel">
            <div className="system-row">
              <div className="system-row-icon"><Server size={18} /></div>
              <div>
                <strong>FastAPI Backend</strong>
                <span>http://127.0.0.1:8000</span>
              </div>
              <span className={backendOnline ? "system-state good" : "system-state bad"}>
                {backendOnline ? "Connected" : "Offline"}
              </span>
            </div>

            <div className="system-row">
              <div className="system-row-icon"><Cpu size={18} /></div>
              <div>
                <strong>Processing Device</strong>
                <span>PyTorch inference acceleration</span>
              </div>
              <span className="system-state neutral">{device}</span>
            </div>

            <div className="system-row">
              <div className="system-row-icon"><Database size={18} /></div>
              <div>
                <strong>Detection and tracking</strong>
                <span>YOLO11s Â· ByteTrack</span>
              </div>
              <span className="system-state good">Ready</span>
            </div>
          </div>
        </section>

        <footer className="footer">
          <span>AI 3rd Vision Tracker</span>
          <span>Powered by YOLO11s Â· FastAPI Â· PyTorch</span>
        </footer>
      </main>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </div>
  );
}

export default App;