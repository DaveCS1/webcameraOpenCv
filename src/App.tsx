import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCcw, FlipHorizontal, Upload, Search } from 'lucide-react';

declare global {
  interface Window {
    cv: any;
  }
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const [image, setImage] = useState<string | null>(null);
  const [templateImage, setTemplateImage] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [isRearCamera, setIsRearCamera] = useState(false);
  const [isOpenCVReady, setIsOpenCVReady] = useState(false);
  const [matchResults, setMatchResults] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [matchThreshold, setMatchThreshold] = useState(0.7);
  const [matchMethod, setMatchMethod] = useState(5); // TM_CCOEFF_NORMED
  const [processedImageWithBoxes, setProcessedImageWithBoxes] = useState<string | null>(null);
  const [liveEdgeDetection, setLiveEdgeDetection] = useState(false);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    // Check if OpenCV is already loaded or being loaded
    if (window.cv || document.querySelector('script[src*="opencv.js"]')) {
      if (window.cv && window.cv.Mat) {
        setIsOpenCVReady(true);
      }
      return;
    }

    // Load OpenCV.js
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.x/opencv.js';
    script.async = true;
    script.onload = () => {
      // Wait for OpenCV to be ready
      const checkOpenCV = () => {
        if (window.cv && window.cv.Mat) {
          setIsOpenCVReady(true);
          console.log('OpenCV.js is ready');
        } else {
          setTimeout(checkOpenCV, 100);
        }
      };
      checkOpenCV();
    };
    document.head.appendChild(script);
  }, []);

  const startCamera = async () => {
    try {
      // Stop any existing stream
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: isRearCamera ? 'environment' : 'user'
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Unable to access camera. Please ensure you have granted camera permissions.");
    }
  };

  const switchCamera = () => {
    setIsRearCamera(!isRearCamera);
    startCamera();
  };

  const takePicture = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        setImage(imageData);
        
        // Stop camera stream
        const stream = videoRef.current.srcObject as MediaStream;
        stream?.getTracks().forEach(track => track.stop());
        setCameraActive(false);
      }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setImage(result);
        setCameraActive(false);
        // Stop any active camera stream
        if (videoRef.current?.srcObject) {
          const stream = videoRef.current.srcObject as MediaStream;
          stream.getTracks().forEach(track => track.stop());
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleTemplateUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setTemplateImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const performTemplateMatching = async () => {
    if (!image || !isOpenCVReady || !templateImage) {
      alert('OpenCV is not ready, no image available, or no template loaded');
      return;
    }

    setIsProcessing(true);
    setMatchResults(null);

    try {
      // Load the captured/uploaded image
      const img = new Image();
      const template = new Image();
      
      img.onload = () => {
        template.onload = () => {
          try {
            // Create canvas for source image
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(img, 0, 0);

            // Create canvas for template image
            const templateCanvas = document.createElement('canvas');
            templateCanvas.width = template.width;
            templateCanvas.height = template.height;
            const templateCtx = templateCanvas.getContext('2d');
            templateCtx?.drawImage(template, 0, 0);

            // Convert to OpenCV Mat
            const src = window.cv.imread(canvas);
            const templ = window.cv.imread(templateCanvas);
            
            console.log('Processing image with OpenCV...');
            console.log('Source dimensions:', src.rows, 'x', src.cols);
            console.log('Template dimensions:', templ.rows, 'x', templ.cols);
            
            // Convert to grayscale for template matching
            const srcGray = new window.cv.Mat();
            const templGray = new window.cv.Mat();
            window.cv.cvtColor(src, srcGray, window.cv.COLOR_RGBA2GRAY);
            window.cv.cvtColor(templ, templGray, window.cv.COLOR_RGBA2GRAY);
            
            // Perform template matching
            const result = new window.cv.Mat();
            window.cv.matchTemplate(srcGray, templGray, result, matchMethod);
            
            // Find the best match location
            const minMaxLoc = window.cv.minMaxLoc(result);
            const maxLoc = minMaxLoc.maxLoc;
            const matchValue = minMaxLoc.maxVal;
            
            console.log('Match confidence:', matchValue);
            console.log('Match location:', maxLoc);
            
            // Create a copy of the original image for drawing
            const displayCanvas = document.createElement('canvas');
            displayCanvas.width = img.width;
            displayCanvas.height = img.height;
            const displayCtx = displayCanvas.getContext('2d');
            displayCtx?.drawImage(img, 0, 0);
            
            if (matchValue > matchThreshold) {
              // Draw bounding box on the image
              if (displayCtx) {
                displayCtx.strokeStyle = '#00ff00'; // Green color
                displayCtx.lineWidth = 3;
                displayCtx.strokeRect(maxLoc.x, maxLoc.y, templ.cols, templ.rows);
                
                // Add confidence text
                displayCtx.fillStyle = '#00ff00';
                displayCtx.font = '16px Arial';
                displayCtx.fillText(
                  `${(matchValue * 100).toFixed(1)}%`, 
                  maxLoc.x, 
                  maxLoc.y - 5
                );
              }
              
              setMatchResults({
                processed: true,
                found: true,
                confidence: (matchValue * 100).toFixed(2),
                location: { x: maxLoc.x, y: maxLoc.y },
                message: `Template found with ${(matchValue * 100).toFixed(2)}% confidence at position (${maxLoc.x}, ${maxLoc.y})`,
                imageSize: { width: src.cols, height: src.rows },
                templateSize: { width: templ.cols, height: templ.rows }
              });
            } else {
              // Still show the best match location with red box for debugging
              if (displayCtx) {
                displayCtx.strokeStyle = '#ff0000'; // Red color
                displayCtx.lineWidth = 2;
                displayCtx.setLineDash([5, 5]); // Dashed line
                displayCtx.strokeRect(maxLoc.x, maxLoc.y, templ.cols, templ.rows);
                
                // Add confidence text
                displayCtx.fillStyle = '#ff0000';
                displayCtx.font = '14px Arial';
                displayCtx.fillText(
                  `${(matchValue * 100).toFixed(1)}%`, 
                  maxLoc.x, 
                  maxLoc.y - 5
                );
              }
              
              setMatchResults({
                processed: true,
                found: false,
                confidence: (matchValue * 100).toFixed(2),
                message: `Template not found. Best match: ${(matchValue * 100).toFixed(2)}% confidence (threshold: ${matchThreshold * 100}%)`,
                imageSize: { width: src.cols, height: src.rows }
              });
            }

            // Set the processed image with bounding boxes
            setProcessedImageWithBoxes(displayCanvas.toDataURL('image/jpeg'));

            // Clean up
            src.delete();
            templ.delete();
            srcGray.delete();
            templGray.delete();
            result.delete();
            setIsProcessing(false);
          } catch (error) {
            console.error('Error in template matching:', error);
            setMatchResults({
              error: 'Error processing template matching: ' + error
            });
            setIsProcessing(false);
          }
        };
        template.src = templateImage;
      };
      img.src = image;
    } catch (error) {
      console.error('Error in template matching:', error);
      setMatchResults({
        error: 'Error processing image: ' + error
      });
      setIsProcessing(false);
    }
  };

  const retake = () => {
    setImage(null);
    setMatchResults(null);
    setProcessedImageWithBoxes(null);
    startCamera();
  };

  const reset = () => {
    setImage(null);
    setTemplateImage(null);
    setMatchResults(null);
    setProcessedImageWithBoxes(null);
    setCameraActive(false);
    // Stop any active camera stream
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-md overflow-hidden">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
              <Camera className="w-8 h-8" />
              Fight Stats Analyzer
            </h1>
            {cameraActive && (
              <button
                onClick={switchCamera}
                className="text-gray-600 hover:text-gray-800 p-2 rounded-full hover:bg-gray-100 transition-colors"
                title={`Switch to ${isRearCamera ? 'front' : 'rear'} camera`}
              >
                <FlipHorizontal className="w-6 h-6" />
              </button>
            )}
          </div>

          <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-6">
            {!image && (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
            )}
            {(image || processedImageWithBoxes) && (
              <img
                src={processedImageWithBoxes || image}
                alt={processedImageWithBoxes ? "Processed with matches" : "Captured or uploaded"}
                className="w-full h-full object-cover"
              />
            )}
            {!cameraActive && !image && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Take a photo or upload an image</p>
                </div>
              </div>
            )}
          </div>

          {/* Control Buttons */}
          <div className="flex flex-wrap justify-center gap-3 mb-6">
            {!cameraActive && !image && (
              <>
                <button
                  onClick={startCamera}
                  className="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 flex items-center gap-2 transition-colors"
                >
                  <Camera className="w-5 h-5" />
                  Start Camera
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 flex items-center gap-2 transition-colors"
                >
                  <Upload className="w-5 h-5" />
                  Upload Image
                </button>
                <button
                  onClick={() => templateInputRef.current?.click()}
                  className="bg-purple-500 text-white px-6 py-3 rounded-lg hover:bg-purple-600 flex items-center gap-2 transition-colors"
                >
                  <Upload className="w-5 h-5" />
                  Load Template
                </button>
              </>
            )}
            
            {cameraActive && (
              <button
                onClick={takePicture}
                className="bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 flex items-center gap-2 transition-colors"
              >
                <Camera className="w-5 h-5" />
                Take Picture
              </button>
            )}
            
            {image && (
              <>
                <button
                  onClick={performTemplateMatching}
                  disabled={!isOpenCVReady || isProcessing || !templateImage}
                  className="bg-purple-500 text-white px-6 py-3 rounded-lg hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                >
                  <Search className="w-5 h-5" />
                  {isProcessing ? 'Processing...' : !templateImage ? 'Load Template First' : 'Analyze Fight Stats'}
                </button>
                <button
                  onClick={retake}
                  className="bg-orange-500 text-white px-6 py-3 rounded-lg hover:bg-orange-600 flex items-center gap-2 transition-colors"
                >
                  <RefreshCcw className="w-5 h-5" />
                  Retake
                </button>
                <button
                  onClick={reset}
                  className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 flex items-center gap-2 transition-colors"
                >
                  Reset
                </button>
              </>
            )}
          </div>

          {/* OpenCV Status */}
          <div className="mb-4">
            <div className="flex flex-wrap gap-2">
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
                isOpenCVReady ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  isOpenCVReady ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
                OpenCV: {isOpenCVReady ? 'Ready' : 'Loading...'}
              </div>
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
                templateImage ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  templateImage ? 'bg-green-500' : 'bg-gray-500'
                }`}></div>
                Template: {templateImage ? 'Loaded' : 'Not loaded'}
              </div>
              {cameraActive && (
                <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
                  liveEdgeDetection ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                }`}>
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    liveEdgeDetection ? 'bg-blue-500' : 'bg-gray-500'
                  }`}></div>
                  Live Detection: {liveEdgeDetection ? 'Active' : 'Inactive'}
                </div>
              )}
            </div>
          </div>

          {/* Template Thumbnail */}
          {templateImage && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Template Image</h3>
              <div className="flex justify-center">
                <img
                  src={templateImage}
                  alt="Template"
                  className="max-w-32 max-h-32 object-contain border border-gray-300 rounded-lg shadow-sm"
                />
              </div>
            </div>
          )}

          {/* OpenCV Parameters */}
          {templateImage && (
            <div className="mb-6 bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-4">Template Matching Parameters</h3>
              
              <div className="space-y-4">
                {/* Match Threshold Slider */}
                <div>
                  <label className="block text-sm text-gray-600 mb-2">
                    Match Threshold: {(matchThreshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={matchThreshold}
                    onChange={(e) => setMatchThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>10%</span>
                    <span>100%</span>
                  </div>
                </div>

                {/* Match Method Selector */}
                <div>
                  <label className="block text-sm text-gray-600 mb-2">
                    Matching Method
                  </label>
                  <select
                    value={matchMethod}
                    onChange={(e) => setMatchMethod(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value={0}>TM_SQDIFF</option>
                    <option value={1}>TM_SQDIFF_NORMED</option>
                    <option value={2}>TM_CCORR</option>
                    <option value={3}>TM_CCORR_NORMED</option>
                    <option value={4}>TM_CCOEFF</option>
                    <option value={5}>TM_CCOEFF_NORMED (Recommended)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    TM_CCOEFF_NORMED is recommended for most use cases
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Results Display */}
          {matchResults && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-2">Analysis Results</h3>
              {matchResults.error ? (
                <div className="text-red-600">{matchResults.error}</div>
              ) : (
                <div className="space-y-2">
                  <p className={matchResults.found ? 'text-green-600' : 'text-orange-600'}>
                    {matchResults.message}
                  </p>
                  {matchResults.confidence && (
                    <p className="text-sm text-gray-600">
                      Match confidence: {matchResults.confidence}%
                    </p>
                  )}
                  {matchResults.location && (
                    <p className="text-sm text-gray-600">
                      Found at: ({matchResults.location.x}, {matchResults.location.y})
                    </p>
                  )}
                  {matchResults.imageSize && (
                    <p className="text-sm text-gray-600">
                      Image processed: {matchResults.imageSize.width} × {matchResults.imageSize.height} pixels
                    </p>
                  )}
                  {matchResults.templateSize && (
                    <p className="text-sm text-gray-600">
                      Template size: {matchResults.templateSize.width} × {matchResults.templateSize.height} pixels
                    </p>
                  )}
                  {processedImageWithBoxes && (
                    <p className="text-sm text-blue-600 mt-2">
                      ✓ Visual feedback shown on image above
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* Hidden template input */}
          <input
            ref={templateInputRef}
            type="file"
            accept="image/*"
            onChange={handleTemplateUpload}
            className="hidden"
          />

          {/* Canvas for OpenCV processing */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      </div>
    </div>
  );
}

export default App;