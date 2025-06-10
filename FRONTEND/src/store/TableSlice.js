import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios";

const initialState = {
  header: [
    // {
    //   key: "title",
    //   label: "Title",
    //   type: "text",
    //   subType: "",
    //   icon_key: "item_type",
    //   customHeaderStyles: {
    //     minWidth: "100px"
    //   }
    // },
    {
      key: "fileName",
      label: "File Name",
      type: "text",
      subType: "",
      icon_key: "item_type",
      customHeaderStyles: {
        minWidth: "100px"
      }
    },
    {
      key: "inputFile",
      label: "Input File",
      type: "text",
      subType: "voice_wave",
      // icon_key: "input_file",
      // onClickEmittToParent: true,
      apiCall:true,
       actionType:"input",
      body_styles: {
        fontSize: 25,
        color: "#669EFF",
      },
      // <iconify-icon icon="flat-color-icons:speaker"></iconify-icon>
      hasIcon: {
        alignment: "right",
        icon_name: "fluent-emoji:play-button",
        input_file_url: "",
        tooltip: "Click to Listen",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 20,
          // padding: 0,
          // margin: 0,
          // position: "absolute",

          // borderRadius: 50,
          // color: "#0560FD",
        },
      },
      noText: true,
      customHeaderStyles: {
        minWidth: "70px",
      },
    },
    {
      key: "model_used",
      label: "Model",
      type: "text",
      subType: "",
      icon_key: "item_type",
      customHeaderStyles: {
        minWidth: "100px"
      }
    },
    // {
    //   key: "fileType",
    //   label: "File Type",
    //   type: "text",
    //   subType: "",
    //   customHeaderStyles: {
    //     minWidth: "70px",
    //   },
    // },
     {
      key: "phraseText",
      label: "Phrase Text",
      type: "text", 
      subType: "",
      customHeaderStyles: {
        minWidth: "100px",
      },
      hasIcon: {
        alignment: "right",
        // icon_name: "quill:link-out",
        input_file_url: "",
        tooltip: "Click to see Full Transcription",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 17,
          // padding: 0,
          // margin: 0,
          // position: "absolute",

          // borderRadius: 50,
          color: "#0560FD",
        },
      }
    },
    {
      key: "Transcription",
      label: "Transcription",
      type: "text",
      subType: "",
      customHeaderStyles: {
        minWidth: "50px",
      },
      openModel: true,
      hasIcon: {
        alignment: "right",
        icon_name: "quill:link-out",
        input_file_url: "",
        tooltip: "Click to see Full Transcription",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 17,
          // padding: 0,
          // margin: 0,
          // position: "absolute",

          // borderRadius: 50,
          color: "#0560FD",
        },
      },
    },

    // {
    //   key: "duration",
    //   label: "Duration",
    //   type: "text",
    //   subType: "",
    //   customHeaderStyles: {
    //     minWidth: "70px",
    //   },
    // },
    {
      key: "outputFile",
      label: "Output Audio",
      type: "text",
      subType: "voice_wave",
      // onClickEmittToParent: true,
       actionType:"output",
      apiCall:true,
      noText: true,
      hasIcon: {
        alignment: "right",
        icon_name: "fluent-emoji:play-button",
        output_file_url: "",
        tooltip: "Click to Listen",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 20,
          // padding: 0,
          // margin: 0,
          // position: "absolute",

          // borderRadius: 50,
          // color: "#0560FD",
        },
      },
      customHeaderStyles: {
        minWidth: "150px",
      },
    },
    // {
    //   key: "dateAndtime",
    //   label: "Date & Time",
    //   type: "dateAndtime",
    //   subType: "",
    //   customHeaderStyles: {
    //     minWidth: "100px",
    //   },
    // },
    {
      key: "accuracy",
      label: "Word Accuracy",
      type: "text",
      subType: "",
      customHeaderStyles: {
        minWidth: "100px",
      },
    },
  ],
  body: [],
  bodyTest: [
    {
      title: "New Recording",

      inputFile: ".mp322",
      fileType: ".mp4",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      audio: {
        url: "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
        type: ".mp3",
      },
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ri:mic-fill",
        styles: {
          backgroundColor: "#5A97FF",
          fontSize: 30,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "Sample Audio",
      inputFile: ".mp3",
      fileType: "",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ic:baseline-upload",
        styles: {
          backgroundColor: "#ff898b",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "New Recording",
      inputFile: ".mp3",
      fileType: ".mp4",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ri:mic-fill",
        styles: {
          backgroundColor: "#5A97FF",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
    },
    {
      title: "Sample Audio",
      inputFile: ".mp3",
      fileType: "",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ic:baseline-upload",
        styles: {
          backgroundColor: "#ff898b",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "New Recording",
      inputFile: ".mp3",
      fileType: ".mp4",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ri:mic-fill",
        styles: {
          backgroundColor: "#5A97FF",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "Sample Audio",
      inputFile: ".mp3",
      fileType: "",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ic:baseline-upload",
        styles: {
          backgroundColor: "#ff898b",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "New Recording",
      inputFile: ".mp3",
      fileType: ".mp4",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ri:mic-fill",
        styles: {
          backgroundColor: "#5A97FF",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
    {
      title: "Sample Audio",
      inputFile: ".mp3",
      fileType: "",
      Transcription: "Hi! its a new audio recording...",
      duration: "00:10",
      "date&time": "23/09/2024  17:40",
      play: "",
      input_file: {
        icon_name: "bi:soundwave",
        input_file_url: "",
        styles: {
          // backgroundColor: "#5A97FF",
          fontSize: 30,
          // padding: 3,
          // borderRadius: 50,
          color: "c3d9ff",
        },
      },
      item_type: {
        icon_name: "ic:baseline-upload",
        styles: {
          backgroundColor: "#ff898b",
          fontSize: 10,
          padding: 3,
          borderRadius: 50,
          color: "#fff",
        },
      },
    },
  ],
  actions: [
    {
      key: "download",
      label: "Download",
      tooltip:
        "Download Transcripted ZIP file (Input Audio, Output Audio, Transcripted Text).",
      icon: "ri:download-fill",
      color: "#0560FD",
    },
    {
      key: "delete",
      label: "Delete",
      icon: "fluent:delete-28-regular",
      color: "red",
      onClickEmittToParent: true,
    },
  ],
};
const TableSlice = createSlice({
  name: "data",
  initialState,
  reducers: {
    updateBody: (state, action) => {
      state.body = action.payload;
    },
    addBodyItem: (state, action) => {
      state.body.push(action.payload);
    },
    // Add more reducers as needed
    removeBodyItem: (state, action) => {
      if (
        window.confirm(
          `Are you sure you want to delete ${action.payload?.title} record?`
        )
      ) {
        state.body.splice(action.payload.internalIndex, 1); // The second parameter specifies the number of elements to remove
        // alert(`Element deleted. Updated array: ${array}`);
      } else {
        alert("Deletion canceled.");
      }

      // console.log(action, state.body)
      // state.body.push(action.payload);
    },
  },
});

export const { updateBody, addBodyItem, removeBodyItem } = TableSlice.actions;
export default TableSlice.reducer;

// extraReducers: (builder) => {
//   builder
//     .addCase(fetchBodyData.pending, (state) => {
//       // Optionally handle loading state
//     })
//     .addCase(fetchBodyData.fulfilled, (state, action) => {
//       state.body = action.payload; // Update body with fetched data
//     })
//     .addCase(fetchBodyData.rejected, (state, action) => {
//       console.error('Failed to fetch body data: ', action.error.message);
//     });
// },
