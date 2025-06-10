import React, { useCallback, useEffect, useRef, useState } from "react";
import "./ListingScreen.css"; // Import the CSS file for styling
// import logoPng from "./../assets/images/logo.png";
import logo from "./../assets/images/logo.png";
import BasicTable from "../ReusableComponents/BasicTable";
import { Icon } from "@iconify/react";
import { useDispatch, useSelector } from "react-redux";
import {
  Autocomplete,
  Avatar,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Modal,
  Popover,
  TextField,
  Typography,
} from "@mui/material";
import AudioRecorder from "../ReusableComponents/AudioRecorder";
import {
  addBodyItem,
  fetchBodyData,
  removeBodyItem,
} from "../store/TableSlice";
import axios from "axios";
import AudioLoader from "../ReusableComponents/AudioLoaders/AudioLoader";
import ModalHeader from "../ReusableComponents/ModelHeader";
import JSZip from "jszip";
import { saveAs } from "file-saver";
import { signInWithPopup, signOut } from "firebase/auth";
import { auth, provider } from "../firebaseConfig";
import { useNavigate } from "react-router-dom";
import { ConfirmationDialog } from "../ReusableComponents/ConfirmationDialog";
import { BASEURL } from "../ReusableComponents/BaseURL";
import { PhraseUploadDialog } from "../ReusableComponents/PhraseUploadDialog";

function ListingScreen() {
  const APIURL = window?.location.hostname;
  let protocol = window.location.protocol;
  const CryptoJS = require("crypto-js");
  const navigate = useNavigate();
  const secureKey = process.env.REACT_APP_SECURE_KEY;
  const [respDatastate, setrespDatastate] = useState(null);
  const [active, setActive] = useState("upload");
  const [isRecording, setIsRecording] = useState(false);
  const [metaData, setMetaData] = useState(null);
  // const [showDialog, setShowDialog] = useState(false);
  const [bodys, setBodys] = useState([]);
  const [file, setFile] = useState(null);
  const [pharseFile, setPharseFile] = useState(null);
  const [fileMetData, setFileMetaData] = useState({});
  const [loader, setLoader] = useState(false);
  // const [inputPlayerModal, setInputPlayerModal] = useState(false);
  // const [outputPlayerModal, setOutputPlayerModal] = useState(false);
  // const [inputPlayerURL, setInputPlayerURL] = useState("");
  // const [outputPlayerURL, setOutputPlayerURL] = useState("");
  const [transcriptionProcessData, setTranscriptionProcessData] =
    useState(null);
  console.log(transcriptionProcessData, "transcriptionProcessData");
  const [newBodyItem, setNewBodyItem] = useState({});
  const [open, setOpen] = useState(false);
  const [logoutDailog, setLogoutDailog] = useState(false);
  const [showPhraseUploadDialog, setShowPhraseUploadDialog] = useState(false);
  const [phraseTextOpen, setPhraseTextOpen] = useState(false);
  const [showUploadScreen, setShowUploadScreen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [deletePayload, setDeletePayload] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [audioTime, setAudioTime] = useState(null);
  const [dailogActions, setDailogActions] = useState(true);
  const [downloadLoad, setDownloadLoad] = useState(false);

  const handleToggle = (option) => {
    setActive(option);
  };

  const [modelOptions, setModelOptions] = useState([
    // {
    //   id:"",
    //   label: "whisper-small",
    // },
    // {
    //   id:"",
    //   label: "whisper-small-finetuned-on-m01",
    // },
  ]);
  console.log(modelOptions, "modelOptions");
  const [model, setModel] = useState(null);
  const currentAudioRef = useRef(null);

  const getModelListApi = async () => {
    try {
      const response = await axios.get(BASEURL + `get_available_models`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      console.log("response", response?.data?.models, response);
      setModelOptions(response?.data?.models);
      setModel(response?.data?.models[0]);
      // return response.data;
    } catch (error) {
      console.error("Error during API call:", error);
       throw error;
    }
  };
  useEffect(() => {
    getModelListApi();
  }, []);
  const handleAudioPlay = (event) => {
    // If there's already an audio playing, pause it
    if (currentAudioRef.current && currentAudioRef.current !== event.target) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0; // Reset time if you want
    }

    // Update the reference to the currently playing audio
    currentAudioRef.current = event.target;
  };
  const ResetDefault = () => {
    setIsRecording(false);
  };
  function CircularProgressWithLabel(props) {
    return (
      <Box
        sx={{
          position: "fixed", // Use 'fixed' to center it in the viewport
          top: "50%", // Move down 50% of the viewport height
          left: "50%", // Move right 50% of the viewport width
          transform: "translate(-50%, -50%)", // Adjust back by half its own size
          display: "inline-flex",
        }}
      >
        <CircularProgress variant="determinate" {...props} size={100} />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: "absolute",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Typography variant="h6" component="div" sx={{ color: "black" }}>
            {`${Math.round(props.value)}%`}
          </Typography>
        </Box>
      </Box>
    );
  }

  const getRecordTime = (time) => {
    setAudioTime(time);
  };
  const getUserDetails = () => {
    const userEncrypted = localStorage.getItem("user");

    if (userEncrypted) {
      try {
        const decryptedData = CryptoJS.AES.decrypt(
          userEncrypted,
          secureKey
        ).toString(CryptoJS.enc.Utf8);
        if (!decryptedData) {
          console.error("Decrypted data is empty.");
          return null;
        }
        const userDetails = JSON.parse(decryptedData);
        return userDetails;
      } catch (error) {
        console.error("Error during decryption or parsing:", error);
        return null;
      }
    } else {
      return null; // Return null if no user data exists
    }
  };
  const userDetails = getUserDetails();
  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(secs).padStart(2, "0")}`;
  };

  const handleFile = (file) => {
    // Here you can add any validation or handling logic for the file

    document.getElementById("fileInput").value = "";

    if (file.size <= 50 * 1024 * 1024) {
      // setLoader(true);

      // Check for file size limit (200MB)
      // if (!userDetails) {
      //   setShowDialog(true);
      // } else {
      // }
      // handleSubmit(file);
      if (!showUploadScreen) {
        setFile(file);
        //  setPhraseTextOpen(false)
      } else {
        setPharseFile(file);
      }
    } else {
      alert("File size exceeds the limit of 50MB");
    }
  };
  console.log(file, "file");

  const handleDrop = (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };
  // const handleFileUploadClick = (event) => {
  //   if (!userDetails) {
  //     event.preventDefault();
  //     setShowDialog(true);
  //     console.log(userDetails, "userDetails");
  //   } else {
  //     if (!file) document.getElementById("fileInput").click(); // Trigger file input click
  //   }
  // };
  const handleFileInputChange = (event, type) => {
    const files = event.target.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const metaInformation = {
    requiredSerialNumber: true,
    paginatedSerialNumber: false,
    paginationMetaData: {
      count: 10,
      page: 1,
    },
  };

  const fetchData = async () => {
    try {
      const data = await getData("get_user_records/" + userDetails?.user_id);
      setBodys(data?.user_records);
    } catch (error) {
      console.error("Error fetching user records:", error);
    }
  };

  let inputAudio;
  let outputAudio;
  const handleDownload = async (dataInput) => {
    try {
      setDownloadLoad(true);
      const zip = new JSZip();

      // Add text content
      const textContent = dataInput?.Transcription;
      zip.file("transcription.txt", textContent);
      inputAudio = await getData("temp_url?fileName=" + dataInput?.inputFile);
      outputAudio = await getData("temp_url?fileName=" + dataInput?.outputFile);

      // Add audio files (Assuming the audio files are in the `public` folder or URLs)

      const audio1Url = inputAudio?.temp_URL; // Replace with your actual path
      const audio2Url = outputAudio?.temp_URL; // Replace with your actual path

      // Fetch audio files as blobs
      const audio1Blob = await fetch(audio1Url).then((res) => res.blob());
      const audio2Blob = await fetch(audio2Url).then((res) => res.blob());

      let newAudio = new Audio(dataInput);
      // Add audio files to the ZIP
      zip.file("input_file.mp3", audio1Blob);
      zip.file("transcripted_output.mp3", audio2Blob);

      // Generate the ZIP file and trigger download
      zip.generateAsync({ type: "blob" }).then((content) => {
        saveAs(content, dataInput?.title || "Download" + ".zip");
      });
    } catch (error) {
      console.error(error);
    } finally {
      setDownloadLoad(false);
    }
  };
  let previousAudioUrl = null;

  async function fetchAudioAsBlob(url) {
    try {
      // Clear previous audio URL if it exists
      if (previousAudioUrl) {
        URL.revokeObjectURL(previousAudioUrl);
      }

      // Fetch the audio file from the URL
      const response = await fetch(url);

      // Check if the response is okay
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      // Convert the response to a Blob
      const audioBlob = await response.blob();

      // Create a URL for the Blob
      const audioUrl = URL.createObjectURL(audioBlob);

      // Store the new audio URL
      previousAudioUrl = audioUrl;

      // You can return the audio URL or the Blob, depending on your needs
      return {
        audioBlob,
        audioUrl,
      };
    } catch (error) {
      console.error("Error fetching audio:", error);
    }
  }
  const loginAPI = async (payload) => {
    try {
      const response = await axios.post(BASEURL + `store_user`, payload, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      return response.data;
    } catch (error) {
      console.log("Error during API call:", error);
      // throw error;
    }
  };
  // const handleLogin = async () => {
  //   try {
  //     const result = await signInWithPopup(auth, provider);
  //     const data = result?.user;

  //     // You can access user details here
  //     // const { email, displayName, photoURL, uid } = user;
  //     const user = data?.providerData[0];
  //     // Optionally call your API
  //     const response = await loginAPI(user);
  //     if (response?.stored_data?.user_id) {
  //       // navigate("/Home");
  //       const encryptedData = CryptoJS.AES.encrypt(
  //         JSON.stringify(response?.stored_data),
  //         secureKey
  //       ).toString();
  //       localStorage.setItem("user", encryptedData);
  //     }
  //   } catch (error) {
  //     console.error("Error during login:", error);
  //   } finally {
  //   }

  // };
  const handleOptions = () => {
    setOpenDialog(true);
  };
  const handleClosePopover = () => {
    setOpenDialog(false);
  };
  const handleConfirmlogout = () => {
    handleLogout();
    setLogoutDailog(false);
  };
  const handleCancellogout = () => {
    setLogoutDailog(false);
  };
  const handleOpenlogout = () => {
    setLogoutDailog(true);
  };
  const handleLogout = async () => {
    await signOut(auth);
    localStorage.clear();
    navigate("/");
  };

  const handlePhraseText = (val) => {
    if (val === "Confirm") {
      setShowPhraseUploadDialog(true); // Open phrase text upload dialog
    } else {
      // Skip phrase text upload and process audio file directly
      processAudioUpload(file, null);
    }
    setPhraseTextOpen(false); // Close the confirmation dialog
  };

  const handlePhraseTextUpload = (pharseFile) => {
    if (pharseFile) {
      processAudioUpload(file, pharseFile); // Call API with both audio and phrase file
      setShowPhraseUploadDialog(false); // Close upload dialog
    } else {
      alert("Please upload a phrase text file."); // Ensure a file is uploaded
    }
  };

  // Allow user to cancel the phrase text upload even after pressing "Yes"
  const cancelPhraseTextUpload = () => {
    setShowPhraseUploadDialog(false); // Close the dialog
    setPharseFile(null); // Clear uploaded phrase file
    processAudioUpload(file, null); // Proceed with audio upload only
  };

  // Handle file input for phrase text
  const handlePhraseFileInput = (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      setPharseFile(files[0]);
    }
  };

  useEffect(() => {
    if (userDetails?.user_id) {
      fetchData();
    }
  }, []);
  useEffect(() => {
    if (file && active === "upload") {
      setPhraseTextOpen(true);
    }
  }, [file]);
  let fileName;
  let duration;
  let newAudioFile;
  let respData;

  const processAudioUpload = async (audioFile, phraseFile) => {
    newAudioFile = audioFile?.recordedURL ? audioFile.recordedURL : audioFile;

    if (audioFile?.recordedURL) {
      const audio1Blob = await fetch(audioFile?.recordedURL).then((res) =>
        res.blob()
      );
      newAudioFile = audio1Blob;
      setFile(audio1Blob);
    }

    // Set the file name based on the input
    fileName = audioFile?.fileName
      ? audioFile.fileName
      : "NewFile_" + new Date().getTime();

    const audio = new Audio(URL.createObjectURL(newAudioFile));
    audio.onloadedmetadata = () => {
      duration = Math.floor(audio.duration); // Get the duration in seconds
    };

    // Wait for audio metadata to load
    await new Promise((resolve) => {
      audio.onloadedmetadata = () => {
        duration = Math.floor(audio.duration);
        resolve();
      };
    });

    const formData = new FormData();
    if (phraseFile) {
      formData.append("text_file", phraseFile); // Add the phrase text file
    }
    formData.append("audio", newAudioFile);
    formData.append("type", "input");
    formData.append("user_id", userDetails?.user_id);
    formData.append("model", model?.id);
    try {
      setLoader(true);
      const response = await axios.post(BASEURL + `process_audio`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      respData = { ...response?.data };

      if (respData?.generated_speech_url) {
        respData["play"] = respData?.generated_speech_url;
        const response = await getData(
          "temp_url?fileName=" + respData?.generated_speech_url
        );
        respData["play_url"] = response.temp_URL;
      }

      if (respData?.input_audio_url) {
        respData["inputFile"] = respData?.input_audio_url;
        const response = await getData(
          "temp_url?fileName=" + respData?.input_audio_url
        );
        respData["inputFile_url"] = response.temp_URL;
      }

      setMetaData({
        newAudioFile: newAudioFile?.type,
        duration: audioTime ? formatTime(audioTime) : formatTime(duration),
        fileName: fileName,
      });
      setrespDatastate(respData);
      if (respData?.generated_speech_url) {
        setTranscriptionProcessData(respData);

        setTranscriptionProcessData((prev) => ({
          ...prev,
          generated_speech_url: respData?.play_url,
        }));
        metaDataApi();
        setNewBodyItem({
          inputFile: fileName,
          fileType: newAudioFile.type,
          Transcription:
            response.data.transcription || "Transcription not available",
          duration: formatTime(duration) || "-",
          dateAndtime: new Date().toLocaleString(),
          outputFile: respData?.play,
          output_file: {
            icon_name: "bi:soundwave",
            // output_file_url: respData?.generated_speech_url,
            styles: {
              color: "c3d9ff",
              fontSize: 15,
            },
          },
          input_file: {
            icon_name: "bi:soundwave",
            // input_file_url: file ? URL.createObjectURL(file) : "",
            styles: {
              color: "c3d9ff",
              fontSize: 15,
            },
          },
          item_type: audioFile?.recordedURL
            ? {
                icon_name: "ri:mic-fill",
                styles: {
                  backgroundColor: "#5A97FF",
                  fontSize: 15,
                  padding: 3,
                  borderRadius: 50,
                  color: "#fff",
                },
              }
            : {
                icon_name: "ic:baseline-upload",
                styles: {
                  backgroundColor: "#ff898b",
                  fontSize: 15,
                  padding: 3,
                  borderRadius: 50,
                  color: "#fff",
                },
              },
        });

        // handleOpen("View");
        setIsRecording(false);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoader(false);
      fetchData();
      // transcriptionProcessData &&  metaDataApi();
    }
  };
  const metaDataApi = useCallback(async () => {
    const payload = {
      inputFile: transcriptionProcessData?.inputFile,
      duration: metaData?.duration,
      fileType: metaData?.newAudioFile,
      dateAndtime: new Date().toLocaleString(),
      outputFile: transcriptionProcessData?.play,
      Transcription: transcriptionProcessData?.transcription,
      user_id: transcriptionProcessData?.user_id,
    };

    try {
      const response = await axios.post(BASEURL + "save_metadata", payload);
      fetchData(); // Ensure fetchData is defined in your component
    } catch (error) {
      console.error(error);
    }
  }, [transcriptionProcessData?.user_id, metaData]); // Include dependencies

  // useEffect(() => {
  //   if (active) { // Check if active is true before calling the API
  //     metaDataApi();
  //   }
  // }, [metaDataApi, active]);

  const getData = async (url) => {
    try {
      const response = await axios.get(BASEURL + url);
      return response.data; // Return the data from the response
    } catch (error) {
      console.log("Error posting data:", error);
      // throw error; // Rethrow the error for further handling
    }
  };
  const handleOutputAudio = async () => {
    // console.log(transcriptionProcessData,"transcriptionProcessData");
    // const res  = await getData('temp_url?fileName=' + transcriptionProcessData?.inputFile)
    // return res?.temp_URL;
  };
  const handleSubmit = async (audioFile) => {
    // event.preventDefault();

    let newAudioFile = audioFile?.recordedURL
      ? audioFile.recordedURL
      : audioFile;

    if (audioFile?.recordedURL) {
      const audio1Blob = await fetch(audioFile?.recordedURL).then((res) =>
        res.blob()
      );
      newAudioFile = audio1Blob;
      setFile(audio1Blob);
    }
    if (newAudioFile) {
      await processAudioUpload(newAudioFile); // Proceed with the upload if user is logged in
    } else {
      alert("Please select an audio file");
    }
  };
  const disguardApi = async () => {
    const payload = {
      files: [
        transcriptionProcessData?.inputFile,
        transcriptionProcessData?.play,
      ],
    };
    const res = await axios.post(BASEURL + "remove_audio_s3", payload);
    fetchData();
  };
  const handleDeleteCancel = () => {
    setDeleteConfirm(false);
  };
  const handleDelete = (payload) => {
    deleteApi(payload);
    setDeleteConfirm(false);
  };
  const handleDeleteConfirm = () => {
    // setDeletePayload(true)
    handleDelete(deletePayload);
  };
  const deleteApi = async (payload) => {
    const res = await axios.post(BASEURL + "remove_record", payload);
    fetchData();
  };
  const handleClose = (type) => {
    setOpen(!open);
    if (type === "clear") disguardApi();
  };

  // const handleInputModalClose = () => {
  //   setInputPlayerModal(false);
  // };
  // const handleOutputModalClose = () => {
  //   setOutputPlayerModal(false);
  // };
  const handleOpen = (type) => {
    setOpen(true);
    if (type === "View") {
      setDailogActions(true);
    }
  };
  const { header, body, actions } = useSelector((state) => state.data);
  const dispatch = useDispatch();

  const handleFileNameClick = () => {
    setIsEditing(true);
  };

  const handleBlur = () => {
    setIsEditing(false);
  };

  const handleChange = (e) => {
    setFileMetaData((prev) => {
      return { ...prev, ...{ fileName: e.target.value } };
    });
    // setFileName(e.target.value);
  };

  const actionEmitter = async (e) => {
    if (e?.action?.key === "download") {
      handleDownload(e?.item);
    } else if (e?.action?.key === "delete") {
      // handleDownload(e?.item)
      // deleteApi(e?.item)

      setDeleteConfirm(true);
      setDeletePayload(e?.item);
      // console.log(e?.item,"muzz");

      // dispatch(removeBodyItem(e?.item));
    }
    // else if (e?.action?.type === "input") {

    // inputAudio()
    // setInputPlayerModal(true);
    // console.log(e?.data,"rammmmmm");

    // open a modal to play input value
    // }
    // else if (e?.action?.type === "output") {
    //   setOutputPlayerModal(true);
    //   async function call (){

    //     const output_audio_url = await getData('http://192.168.1.81:5050/temp_url?fileName=' + e?.data?.outputFile)
    //      setOutputPlayerURL(output_audio_url?.temp_URL);
    //   }
    //   call()
    //   // open a modal to play input value
    //   // console.log(e?.data,"rammmmmm");
    // }
    else if (e?.action?.type === "openModel") {
      let input_audio_url;
      let output_audio_url;

      input_audio_url = await getData(
        "temp_url?fileName=" + e?.data?.inputFile
      );
      output_audio_url = await getData(
        "temp_url?fileName=" + e?.data?.outputFile
      );
      setDailogActions(false);
      setTranscriptionProcessData({
        transcription: e?.data?.Transcription,
        generated_speech_url: output_audio_url?.temp_URL,
        input_audio: input_audio_url?.temp_URL,
        duration: e?.data?.duration,
        phraseText: e?.data?.phraseText,
        fileName: e?.data?.inputFile,
      });

      setFileMetaData({
        fileName: e?.data?.inputFile,
        duration: e?.data?.duration,
      });
      setFile(null);
      handleOpen();
    }
  };
  const PhraseTextDialog = () => {
    const handleFileUpload = async (event) => {
      const file = event.target.files[0];
      setPharseFile(file);
      if (file) {
        try {
          await handlePhraseTextUpload(file); // Call the API with the file directly
        } catch (error) {
          console.error("Error uploading phrase text file:", error);
        }
      }
    };

    return (
      <PhraseUploadDialog
        open={showPhraseUploadDialog}
        dialogTitle="Upload Phrase Text"
        dialogMessage="Please upload your phrase text file."
        handleClose={cancelPhraseTextUpload} // Handle the "X" button
        uploadContent={
          <>
            <div className="flex-center">
              <Icon
                icon="solar:cloud-upload-outline"
                style={{ fontSize: "65px", color: "#0560FD" }}
              />
            </div>
            <div className="flex-center">
              <label
                className="browseButtonStyle"
                htmlFor="phraseFileInput"
                style={{ cursor: "pointer" }}
              >
                Browse Phrase Text
              </label>
              <input
                id="phraseFileInput"
                type="file"
                onChange={handleFileUpload} // Handle upload directly
                style={{ display: "none" }}
                accept=".pdf,.doc,.docx,.xls,.xlsx,.txt"
              />
            </div>
          </>
        }
      />
    );
  };

  return (
    <div className="container">
      <div className="top-section">
        <div className="flex-properties-corner">
          <img src={logo} alt="Logo" className="logo" />
          {/* <Avatar 
        src={logo} 
        alt={"User Avatar"}
        sx={{
          width: 45,
          height: 45,
          cursor: "pointer",
          marginTop: "7px",
        }}
        /> */}
          <p className="headerTitle">
            Speech Transcription and Real-Time Processing
          </p>

          <Avatar
            src={userDetails?.photoURL}
            alt={userDetails?.displayName || "User Avatar"}
            onClick={() => handleOptions()}
            sx={{
              width: 45,
              height: 45,
              cursor: "pointer",
              marginTop: "7px",
            }}
          />
          <Popover
            open={openDialog}
            anchorOrigin={{
              vertical: "top",
              horizontal: "right",
            }}
            transformOrigin={{
              vertical: "top",
              horizontal: "bottom",
            }}
            onClose={handleClosePopover}
          >
            {/* Cancel Icon */}
            {/* <div
    style={{ width: "100%", display: "flex", justifyContent: "flex-end" }}
  >
    <Icon
      size="small"
      sx={{
        color: "black",
      }}
      cursor="pointer"
      width={25}
      height={25}
      icon="material-symbols:cancel"
      onClick={() => {
        setOpenDialog(false);
      }}
    />
  </div> */}

            {/* Content Section */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center", // Center content vertically
                textAlign: "center", // Center text
                padding: "16px", // Add padding for better spacing
              }}
            >
              <Avatar
                src={userDetails?.photoURL}
                alt={userDetails?.displayName || "User Avatar"}
                sx={{ width: 60, height: 60 }}
              />
              <Typography variant="body1" sx={{ marginTop: 2 }}>
                {userDetails?.email}
              </Typography>

              {/* Logout Button */}
              <Button
                // size="small"
                variant="contained"
                color="error"
                onClick={handleOpenlogout}
                sx={{ marginTop: 2, textTransform: "none" }} // Space above the button
              >
                Logout
              </Button>
            </div>
          </Popover>
        </div>
          <p className="headerTitle-mobile">Speech Transcription and Real-Time Processing</p>
        <div className="headerContent">
          {/* <p className="headerTitle">
            Speech Transcription and Real-Time Processing
          </p> */}
          <div className="centerRuler"></div>

          <div className="recordSectionContainer">
            <div style={{ width: "60%" }}>
              {/* <Autocomplete
                options={modelOptions.map((option) => option.id)} // Extract names
                value={model}
                onChange={(event, value,option) => {setModel(value) ;console.log(value,"value", event?.target?.value,option  )}} // Get the selected value
                renderInput={(params) => (
                  <TextField {...params} label="Select Model" />
                )}
                disableClearable
              /> */}
              <Autocomplete
                options={modelOptions} // Pass the full array of options
                value={model} // This should hold the selected object, not just a string
                defaultValue={modelOptions[0]}
                onChange={(event, value) => {
                  setModel(value); // The whole option object is available as `value`
                  console.log(value, "Selected Option"); // Log the selected option object
                }}
                getOptionLabel={(option) => option.label || ""} // Display the label in the dropdown
                renderInput={(params) => (
                  <TextField {...params} label="Select Model" />
                )}
                disableClearable
              />
            </div>
            <div className="childSection">
              <p className="headerSubTitle">What would you like to do?</p>

              <div className="slider-container">
                <div
                  className={`slider ${active}`}
                  onClick={() =>
                    handleToggle(active === "upload" ? "record" : "upload")
                  }
                >
                  <div className="toggle" />
                </div>
                <div className="options">
                  <div
                    className={`option ${active === "upload" ? "active" : ""}`}
                    onClick={() => {
                      handleToggle("upload");
                      setIsRecording(false);
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Icon
                        icon="jam:upload"
                        width="30"
                        height="30"
                        style={{ marginRight: "8px" }}
                      />

                      <span style={{ fontSize: "16px" }}>Upload</span>
                    </div>
                  </div>
                  <div
                    className={`option ${active === "record" ? "active" : ""}`}
                    onClick={() => {
                      handleToggle("record");
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Icon
                        icon="material-symbols:mic"
                        width="28"
                        height="28"
                        style={{ marginRight: "8px" }}
                      />
                      <span style={{ fontSize: "16px" }}>Record</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="childSection uploadSection">
              {active === "upload" ? (
                // Your component or JSX for the "upload" condition
                <>
                  <div
                    onDragEnter={preventDefaults}
                    onDragLeave={preventDefaults}
                    onDragOver={preventDefaults}
                    onDrop={handleDrop}
                    className="dropzone"
                  >
                    <span style={{ textAlign: "center" }}>
                      <div className="uploadIconHolder">
                        <Icon
                          icon="solar:cloud-upload-outline"
                          style={{ fontSize: "65px", color: "#0560FD" }}
                        />
                      </div>

                      <div className="uploaderContentSection">
                        Drag & drop files or{" "}
                        <label
                          className="browseButtonStyle"
                          htmlFor="fileInput"
                          style={{ cursor: "pointer" }}
                        >
                          Browse Audio File
                        </label>
                        <input
                          id="fileInput"
                          type="file"
                          onChange={handleFileInputChange}
                          style={{ display: "none" }}
                          accept=".wav,.mp3,.m4a,.mp4"
                        />
                      </div>

                      <span className="uploaderMutedText">
                        Limit 50MB per file. Supported formats: .wav, .mp3,
                        .m4a, .mp4
                      </span>
                    </span>
                  </div>

                  {showPhraseUploadDialog && <PhraseTextDialog />}
                </>
              ) : (
                <>
                  {/* {isRecording ? ( */}
                  <>
                    <AudioRecorder
                      handleSubmit={handleSubmit}
                      ResetDefault={ResetDefault}
                      getRecordTime={getRecordTime}
                      handleOutputAudio={handleOutputAudio}
                      metaDataApi={metaDataApi}
                      respData={respDatastate}
                    />
                  </>
                  {/* ) 
                  : (
                    <span
                      className="recordSectionContainer recordL1"
                      onClick={() => setIsRecording(true)}
                      style={{ cursor: "pointer" }}
                    >
                      <div className="recordSectionL1IconHolder">
                        <Icon
                          icon="material-symbols:mic"
                          style={{ fontSize: "50px", color: "white" }}
                        ></Icon>
                      </div>
                      <div className="titleSection">
                        <p className="titleContainer">Record Speech</p>
                        <p className="mutedText">
                          To record your Speech click on the blue button.
                        </p>
                      </div>
                    </span>
                  )
                  } */}
                </>

                // <Grid2
                //   container
                //   spacing={2}
                //   sx={{
                //     display: "flex",
                //     justifyContent: "space-between",
                //     alignItems: "center",
                //     width: "40%",
                //     padding: "16px", // Added padding for spacing
                //     backgroundColor: "#F3F4F6", // Background color similar to the image
                //     border: "2px solid #E0E0E0", // Light gray border
                //     borderRadius: "16px", // Adjusted for rounded corners
                //     boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)", // Soft shadow
                //   }}
                // >
                //   <Grid2 item size={2}>
                //     <div
                //       style={{
                //         borderRadius: "50%",
                //         borderWidth: "2px",
                //         borderColor: "blue", // Change this to any color you prefer
                //         overflow: "hidden",
                //       }}
                //     >
                //       <Icon
                //         icon="material-symbols:mic"
                //         style={{ fontSize: "65px", color: "white" }}
                //       ></Icon>
                //     </div>
                //   </Grid2>
                //   <Grid2 item size={10} sx={{ color: "#8E8E8E" }}>
                //     <Typography variant="h6">Record Speech</Typography>
                //     <Typography>To record your Speech,</Typography>
                //     <Typography style={{ marginTop: "8px" }}>
                //       click on the <span>blue button</span>.
                //     </Typography>
                //   </Grid2>
                // </Grid2>
              )}
            </div>
          </div>
        </div>

        {/* This section will have the background image */}
      </div>
      <div className="bottom-section">
        {/* The remaining 60% section */}
        {/* <p>This is the bottom section.</p> */}

        <BasicTable
          header={header}
          body={bodys}
          actions={actions}
          metaData={metaInformation}
          actionEmitter={actionEmitter}
          // inputAudio={inputAudio}
        />
      </div>
      {loader && <AudioLoader />}
      {downloadLoad && <AudioLoader />}
      <Dialog
        fullWidth
        maxWidth={false}
        open={open}
        PaperProps={{
          sx: { maxWidth: 720, borderRadius: "15px" },
        }}
      >
        <ModalHeader heading="Audio Transcription" handleClose={handleClose} />
        <DialogContent>
          {transcriptionProcessData && (
            <>
              <p
                style={{ fontSize: "16px", fontWeight: 500, color: "#646464" }}
              >
                Input Audio
              </p>
              <audio
                src={
                  file
                    ? URL.createObjectURL(file)
                    : transcriptionProcessData?.input_audio
                }
                controls
                style={{
                  backgroundColor: "transparent",
                  height: "30px",
                  width: "100%",
                  outline: "none",
                }}
                onPlay={handleAudioPlay}
              />
              <div className="flexProperties" style={{ justifyContent: "end" }}>
                <div className="flexProperties modal_text"></div>
              </div>

              {transcriptionProcessData?.phraseText?.length > 0 && (
                <>
                  {" "}
                  <div
                    className="flexProperties"
                    style={{
                      justifyContent: "space-between",
                      marginTop: "25px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "16px",
                        fontWeight: 500,
                        color: "#646464",
                      }}
                    >
                      Phrase Text
                    </span>
                    <span>
                      <Icon
                        icon="solar:chat-dots-linear"
                        style={{ color: "#A7C7FF", fontSize: "20px" }}
                      ></Icon>
                    </span>
                  </div>
                  <p style={{ fontSize: "20px", fontWeight: 600, margin: 0 }}>
                    {transcriptionProcessData?.phraseText}
                  </p>{" "}
                </>
              )}
              <div
                className="flexProperties"
                style={{ justifyContent: "space-between", marginTop: "25px" }}
              >
                <span
                  style={{
                    fontSize: "16px",
                    fontWeight: 500,
                    color: "#646464",
                  }}
                >
                  Transcription
                </span>
                <span>
                  <Icon
                    icon="solar:chat-dots-linear"
                    style={{ color: "#A7C7FF", fontSize: "20px" }}
                  ></Icon>
                </span>
              </div>
              <p style={{ fontSize: "20px", fontWeight: 600, margin: 0 }}>
                {transcriptionProcessData?.transcription}
              </p>
              <p
                style={{
                  fontSize: "16px",
                  fontWeight: 500,
                  color: "#646464",
                  marginTop: "25px",
                }}
              >
                {" "}
                Transcribed Audio
              </p>
              <audio
                src={transcriptionProcessData?.generated_speech_url}
                controls
                style={{
                  backgroundColor: "transparent",
                  height: "30px",
                  width: "100%",
                  outline: "none",
                }}
                onPlay={handleAudioPlay}
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          {/* {dailogActions ? (
            <>
              <Button
                variant="contained"
                sx={{ backgroundColor: "#303030" }}
                onClick={() => {
                  handleClose("clear");
                }}
              >
                Discard
              </Button>
              <Button variant="contained" onClick={() => StoreData()}>
                Save
              </Button>
            </>
          ) : 
          ( */}
          <Button
            variant="contained"
            sx={{ backgroundColor: "#303030" }}
            onClick={() => {
              handleClose();
            }}
          >
            Close
          </Button>
          {/* // )} */}
        </DialogActions>
      </Dialog>

      {/* <Dialog
        PaperProps={{
          sx: { maxWidth: 720, borderRadius: "15px" },
        }}
        open={showDialog}
        onClose={() => setShowDialog(false)}
      >
        <DialogTitle>Login Required</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Hi there! You need to log in before proceeding.
          </Typography>
        </DialogContent>
        <dailogActions>
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "flex-end",
              marginRight: 10,
              marginBottom: 10,
            }}
          >
            <Button
              id="loginButton"
              size="small"
              variant="contained"
              onClick={() => handleLogin()}
            >
              Login
            </Button>
          </div>
        </dailogActions>
      </Dialog> */}

      {/* <Modal
        open={inputPlayerModal}
        onClose={handleInputModalClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            width: "100vw",
            position: "fixed",
            top: 0,
            left: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            zIndex: 1000,
          }}
        >
          <Icon
            icon="icon-park-solid:close-one"
            style={{
              position: "absolute",
              top: "10px",
              right: "10px",
              cursor: "pointer",
              fontSize: "24px", // Adjust icon size
              color: "#fff", // Change color as needed
            }}
            onClick={handleInputModalClose} // Close the modal when clicked
          />
          <div
            style={{
              border: "0px solid #ccc",
              width: "40%",

              padding: "20px",
              borderRadius: "8px",
          
            }}
          >

            <audio
              src={inputPlayerURL}
              controls
              style={{
                backgroundColor: "transparent",
                height: "50px",
                width: "100%",
                outline: "none",
              }}
            />
          </div>
        </div>
      </Modal> */}
      {/* <Modal
        open={outputPlayerModal}
        onClose={handleOutputModalClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            width: "100vw",
            position: "fixed",
            top: 0,
            left: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            zIndex: 1000,
          }}
        >
          <Icon
            icon="icon-park-solid:close-one"
            style={{
              position: "absolute",
              top: "10px",
              right: "10px",
              cursor: "pointer",
              fontSize: "24px", // Adjust icon size
              color: "#fff", // Change color as needed
            }}
            onClick={handleOutputModalClose} // Close the modal when clicked
          />
          <div
            style={{
              border: "0px solid #ccc",
              width: "40%",

              padding: "20px",
              borderRadius: "8px",

            }}
          >

            <audio
              src={outputPlayerURL}
              controls
              style={{
                backgroundColor: "transparent",
                height: "50px",
                width: "100%",
                outline: "none",
              }}
            />
          </div>
        </div>
      </Modal> */}
      {/* <Dialog
      fullWidth
      maxWidth={false}
      open={logoutDailog}
      PaperProps={{
        sx: { maxWidth: 420, borderRadius: '15px' },
      }}
    >
      <DialogTitle>Confirm Logout</DialogTitle>
      <DialogContent>
        Are you sure you want to log out?
      </DialogContent>
      <DialogActions>
        <Button size="small" onClick={handleCancellogout} color="primary" variant="contained">
          No
        </Button>
        <Button size="small" onClick={handleConfirmlogout} color="error" variant="contained">
          Yes
        </Button>
      </DialogActions>
    </Dialog> */}
      <ConfirmationDialog
        open={logoutDailog}
        dialogTitle="Confirm Logout"
        dialogMessage=" Are you sure you want to log out?"
        cancelButtonText="No"
        confirmButtonText="Yes"
        handleCancel={handleCancellogout}
        handleConfirm={handleConfirmlogout}
      />
      <ConfirmationDialog
        open={deleteConfirm}
        // dialogTitle="Confirm Logout"
        dialogMessage=" Are you sure you want to Delete?"
        cancelButtonText="No"
        confirmButtonText="Yes"
        handleCancel={handleDeleteCancel}
        handleConfirm={handleDeleteConfirm}
      />
      <ConfirmationDialog
        open={phraseTextOpen}
        dialogTitle="Upload Phase Text?"
        dialogMessage="Do you want to upload Phase Text for this file?"
        cancelButtonText="No"
        confirmButtonText="Yes"
        handleCancel={() => handlePhraseText("cancel")}
        handleConfirm={() => handlePhraseText("Confirm")}
      />
    </div>
  );
}
export default ListingScreen;
