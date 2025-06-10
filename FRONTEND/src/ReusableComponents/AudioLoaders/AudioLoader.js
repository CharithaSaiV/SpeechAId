// AudioLoader.js
import React from 'react';
import './AudioLoader.css'; // Import the CSS file for the animation

const AudioLoader = () => {
    return (
        <div className="audio-loader">
            <div className='loader-container'>

                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>

            </div>
        </div>
    );
};

export default AudioLoader;
