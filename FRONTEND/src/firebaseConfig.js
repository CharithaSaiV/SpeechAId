// firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyA5F445DUlXJ2zqrYo1LXecEjLdFKEiXEo",
  authDomain: "speechtranscription-4d4ff.firebaseapp.com",
  projectId: "speechtranscription-4d4ff",
  storageBucket: "speechtranscription-4d4ff.appspot.com",
  messagingSenderId: "1044479725430",
  appId: "1:1044479725430:web:6711c044bfc9fd2ba43457",
  measurementId: "G-L0BPH8FGK7",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

export { auth, provider };
