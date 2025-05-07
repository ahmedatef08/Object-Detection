    // App.jsx
    import React, { useState, useRef, useEffect } from 'react';
    import axios from 'axios';
    import './App.css';

    const colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'];

    function App() {
    const [image, setImage] = useState(null);
    const [detections, setDetections] = useState([]);
    const canvasRef = useRef(null);

    const handleImageChange = (e) => {
        setImage(e.target.files[0]);
        setDetections([]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
        alert("Please upload an image!");
        return;
        }

        const formData = new FormData();
        formData.append('image', image);

        try {
        const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        setDetections(response.data.objects);
        } catch (error) {
        console.error("Error uploading image:", error);
        }
    };

    const processImage = async (method) => {
        if (!image) {
        alert("Please upload an image!");
        return;
        }

        const formData = new FormData();
        formData.append('image', image);
        formData.append('method', method);

        try {
        const response = await axios.post('http://127.0.0.1:5000/process', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            responseType: 'blob'
        });

        const processedUrl = URL.createObjectURL(response.data);
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = processedUrl;
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        } catch (error) {
        console.error("Error processing image:", error);
        }
        };
    useEffect(() => {
    if (image) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.src = URL.createObjectURL(image);
        img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // رسم المربعات لو فيه Detections
        detections.forEach((det, index) => {
            const color = colors[index % colors.length];
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.fillStyle = color;
            ctx.font = "18px Arial";

            const { x, y, width, height } = det.box;
            ctx.strokeRect(x, y, width, height);
            ctx.fillText(`${det.class_name} (${(det.confidence * 100).toFixed(1)}%)`, x, y > 20 ? y - 5 : y + 20);
        });
        };
    }
    }, [image, detections]);

    return (
        <div className="App">
        <h1>Object Detection System</h1>

        <form onSubmit={handleSubmit} className="upload-form">
            <input type="file" accept="image/*" onChange={handleImageChange} />
            <button type="submit">Analyze Image</button>
        </form>

        <div className="buttons">
        <button type="button" onClick={() => processImage("thresholding")}>Thresholding</button>
        <button type="button" onClick={() => processImage("edge")}>Edge-based</button>
        <button type="button" onClick={() => processImage("region")}>Region-based</button>
        <button type="button" onClick={() => processImage("features")}>Feature Extraction</button>
        <button type="button" onClick={() => processImage("clustering")}>Clustering</button>
        </div>


        <div className="canvas-container">
            <canvas ref={canvasRef} width="500" height="500"></canvas>
        </div>

        {detections.length > 0 && (
            <div className="detections">
            <h2>Detections:</h2>
            <ul>
                {detections.map((det, idx) => (
                <li key={idx} style={{ color: colors[idx % colors.length] }}>
                    {det.class_name} - {(det.confidence * 100).toFixed(2)}%
                </li>
                ))}
            </ul>
            </div>
        )}
        </div>
    );
    }

    export default App;
