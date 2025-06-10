import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  Typography,
} from "@mui/material";
import { Icon } from "@iconify/react";
import React, { useState, useRef, useEffect } from "react";
import ModalHeader from "./ModelHeader";

import styles from "./AudioRecorder.module.css";

const AudioRecorder = ({
  handleSubmit,
  ResetDefault,
  getRecordTime,
  handleOutputAudio,
  metaDataApi,
  respData,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [outputAudio, setOutputAudio] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [open, setOpen] = useState(false);
  const [fileName, setFileName] = useState(
    "NewRecording_" + new Date().getTime()
  );
  const [isEditing, setIsEditing] = useState(false);
  

  const handleClose = () => {
    setOpen(!open);
  };

  const inputAudioRef = useRef(null);
  const outputAudioRef = useRef(null);

  const handleInputAudioPlay = () => {
    if (outputAudioRef.current) {
      outputAudioRef.current.pause();
    }
  };

  const handleOutputAudioPlay = () => {
    if (inputAudioRef.current) {
      inputAudioRef.current.pause();
    }
  };

  const handleOpen = () => {
    setOpen(true);
  };
  console.log(respData, "respData");

  useEffect(() => {
    let interval = null;

    if (isRecording) {
      setRecordingTime(0);
      interval = setInterval(() => {
        setRecordingTime((prevTime) => prevTime + 1);
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  useEffect(() => {
    getRecordTime(recordingTime);
  }, [isRecording]);

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(secs).padStart(2, "0")}`;
  };

  const startRecording = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Media recording not supported in this browser.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/mp3" });
        
        setAudioBlob(blob);
        chunksRef.current = [];

        // Now we know we have the blob, so we can call handleSubmit
        try {
          await handleSubmit({
            fileName: fileName,
            recordedURL: URL.createObjectURL(blob), // Use the blob directly instead of audioBlob state
          });
        } catch (error) {
          console.error("Error during submission:", error);
        }
        const res = await handleOutputAudio();
        setOutputAudio(res);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      // The actual submission is now handled in the onstop event
      // No need to check audioBlob here as it won't be ready yet
    }
  };

  const reRecord = () => {
    setAudioBlob(null);
    setIsRecording(false);
  };

  const handleFileNameDoubleClick = () => {
    setIsEditing(true);
  };

  const handleFileNameBlur = (event) => {
    setFileName(event.target.value);
    setIsEditing(false);
  };

  const handleChange = (e) => {
    setFileName(e.target.value);
  };

  return (
    <>
      {audioBlob ? (
        <>
          {!isRecording && (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "30px",
              }}
            >
              <div style={{ textAlign: "center" }}>
                <p>Input Audio</p>
                <audio
                  ref={inputAudioRef}
                  src={URL.createObjectURL(audioBlob)}
                  className={styles.audioPlayer}
                  controls
                  onPlay={handleInputAudioPlay}
                />
              </div>
              <div style={{ textAlign: "center" }}>
                <p>Output Audio</p>
                <audio
                  ref={outputAudioRef}
                  src={respData?.play_url}
                  className={styles.audioPlayer}
                  controls
                  onPlay={handleOutputAudioPlay}
                />
              </div>
            </div>
          )}

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            {isEditing ? (
              <input
                type="text"
                value={fileName}
                onChange={handleChange}
                onBlur={handleFileNameBlur}
                autoFocus
                className={styles.onHoverButton}
                style={{
                  padding: "4px",
                  fontSize: "11px",
                  fontWeight: 500,
                  color: "#646464",
                  margin: 0,
                }}
              />
            ) : (
              !isRecording && (
                <p
                  style={{
                    padding: "4px",
                    fontSize: "11px",
                    fontWeight: 500,
                    color: "#646464",
                    margin: 0,
                    cursor: "pointer",
                  }}
                  className={styles.onHoverButton}
                  onDoubleClick={handleFileNameDoubleClick}
                >
                  File Name: {fileName}
                </p>
              )
            )}
              {isRecording && (
                <div
                style={{
                  display:"flex",
                  justifyContent:'space-between',
                  alignItems:'center',
                  flexDirection:'column',
                }}
                >
                  <p>Recording</p>
                  {formatTime(recordingTime)}
                </div>
              )}
          </div>

           
          {!isRecording ? (
            <Icon
              onClick={startRecording}
              icon="material-symbols:mic"
              width="38"
              height="38"
              style={{
                width: "34px",
                height: "34px",
                borderRadius: "50%",
                color: "#0560FD",
                cursor: "pointer",
                border: "1px solid #0560FD",
                backgroundColor: "transparent",
              }}
            />
          ) : (
            <div
            style={{
              display:'flex',
              justifyContent:'center',
              alignItems:'center',
              paddingTop:50
            }}
            >
            <Icon
              onClick={stopRecording}
              icon="fluent:record-stop-28-regular"
              width="38"
              height="38"
              style={{
                color: "red",
                cursor: "pointer",
                // marginLeft: "10px",
              }}
            />
            </div>
          )}
        </>
      ) : (
        <>
          <div
             style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <p
              style={{
                fontSize: "20px",
                fontWeight: 600,
                marginBottom: 30,
                marginTop: 10,
              }}
            >
              Record
            </p>
            </div>
            <div>
            {isRecording && <div style={{margin:1}}>{formatTime(recordingTime)}</div>}

              <div
                 style={{
                  display: "flex",
                  justifyContent:  !isRecording ? "space-between" :"center",
                  alignItems: "center",
                   marginTop:60,
                   flexDirection:'column'
                }}
              >
                  {!isRecording ? (
                    <Icon
                      onClick={startRecording}
                      icon="material-symbols:mic"
                      style={{
                        width: "34px",
                        height: "34px",
                        borderRadius: "50%",
                        color: "#0560FD",
                        cursor: "pointer",
                        border: "1px solid #0560FD",
                        backgroundColor: "transparent",
                      }}
                    />
                  ) : (
                    <Icon
                      onClick={stopRecording}
                      icon="fluent:record-stop-28-regular"
                      style={{
                        color: "red",
                        cursor: "pointer",
                        width: "34px",
                        height: "34px",
                      }}
                    />
                  )}
                </div>
              </div>
            
        </>
      )}
    </>
  );
};

export default AudioRecorder;
