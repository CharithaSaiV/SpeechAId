import React, { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { GoogleAuthProvider, signInWithPopup, signOut } from "firebase/auth";
import CryptoJS from "crypto-js";
import axios from "axios";
import { auth, provider } from "../firebaseConfig";
import { Button } from "@mui/material";
import styles from './Login.module.css';
import { Icon } from "@iconify/react";

import images from './../assets/images/logo.png'
import facebookIcon from './../assets/images/facebook.png'
import { BASEURL } from "./BaseURL";

const Login = () => {
  const navigate = useNavigate();
  const secureKey = process.env.REACT_APP_SECURE_KEY;
  const provider = new GoogleAuthProvider();
  const handleGoogleLogin = async () => {
    try {
      provider.setCustomParameters({ prompt: 'select_account' });
      const result = await signInWithPopup(auth, provider);
      const data = result?.user;

      const user = data?.providerData[0];
      console.log(user, "user");

      const response = await loginAPI(user);
      if (response?.stored_data?.user_id) {
        const encryptedData = CryptoJS.AES.encrypt(
          JSON.stringify(response?.stored_data),
          secureKey
        ).toString();
        localStorage.setItem("user", encryptedData);
        navigate("/home");
      }
    } catch (error) {
      console.error("Error during login:", error);
    }
  };

  const loginAPI = async (payload) => {
    try {
      const response = await axios.post(
        BASEURL + "store_user",
        payload,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error("Error during API call:", error);
      throw error;
    }
  };

  return (
    <div className={styles.containerParent}>
      {/* Left side with background image */}
      <div className={styles.leftChild}></div>

      {/* Right side with logo, text, and buttons */}

      <div className={styles.rightChild}>
        <img
          src={images} // Replace with the actual logo URL
          alt="Company Logo"
          className={styles.logo}
        />


        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>

          <div style={{ marginTop: '1rem' }}>


            <div className={styles.welcomeText}>Hello, Welcome to</div>
            <div className={styles.projectName}>Speech-Impairment</div>
            <div className={styles.projectDescription}>This Speech Impariment leverages advanced AI models to assist individuals with speech impairments by converting impaired speech into accurate, clear, and understandable transcription and proper speech. The system processes spoken input with speech irregularities and transforms it into structured, grammatically correct text or speech, enhancing communication in real-time.

            </div>
          </div>


          <div className={styles.buttonsContainer}>
            <Button
              className={styles.loginButtonGoogle}
              variant="contained"
              color="primary"
              sx={{ textTransform: 'none' }}

              onClick={handleGoogleLogin}
            >
              <Icon icon="devicon:google" style={{ marginRight: 10 }}></Icon>  Log In with Google            </Button>
            <Button
              className={styles.loginButtonFacebook}
              variant="contained"
              color="secondary"
              sx={{ textTransform: 'none' }}

            >

              <img src={facebookIcon} style={{ width: '15px', height: '15px', marginRight: 10 }}></img> Log In with Facebook            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
