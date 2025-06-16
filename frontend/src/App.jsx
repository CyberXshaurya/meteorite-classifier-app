import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { FaMeteor, FaUpload, FaSearch, FaMobileAlt, FaTabletAlt, FaLaptop } from 'react-icons/fa';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Is the backend server running?');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>
          <FaMeteor className="meteorite-icon" />
          Meteorite Texture Classifier
        </h1>
        <p>Upload an image of a meteorite sample to analyze its mineral composition and texture characteristics.</p>
      </header>

      <div className={`upload-section ${selectedFile ? 'active' : ''}`}>
        <input 
          type="file" 
          id="file-upload" 
          onChange={handleFileChange} 
          accept="image/png, image/jpeg, image/jpg" 
        />
        <div className="upload-buttons">
          <label htmlFor="file-upload" className="custom-file-upload">
            <FaUpload style={{ marginRight: '8px' }} />
            Choose Meteorite Image
          </label>
          <button onClick={handlePredict} disabled={isLoading || !selectedFile}>
            {isLoading ? (
              <span className="loading-message">
                <span className="loading-spinner"></span>
                Analyzing Sample...
              </span>
            ) : (
              <>
                <FaSearch style={{ marginRight: '8px' }} />
                Classify Texture
              </>
            )}
          </button>
        </div>
      </div>

      {error && <p className="error-message">{error}</p>}

      <div className="results-section">
        {preview && !prediction && (
          <div className="image-preview">
            <h3>Sample Preview</h3>
            <img src={preview} alt="Selected meteorite sample" />
          </div>
        )}

        {isLoading && !prediction && (
          <div className="loading-message">
            <span className="loading-spinner"></span>
            Scanning mineral patterns...
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <h2>Analysis Results</h2>
            <img src={prediction.processed_image} alt="Processed meteorite sample" />
            <div className="details">
              <h3>Composition Analysis: <span>{prediction.verdict}</span></h3>
              <p><strong>Metal Content:</strong> {prediction.metal_composition}</p>
              <p><strong>Silicate Matrix:</strong> {prediction.silicate_composition}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;