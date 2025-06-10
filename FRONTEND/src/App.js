import logo from "./logo.svg";
import "./App.css";
import BasicTable from "./ReusableComponents/BasicTable";
import ListingScreen from "./Screens/ListingScreen";
import Login from "./ReusableComponents/Login";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { lazy, Suspense } from "react";

function App() {
  const Login = lazy(() => import(".././src/ReusableComponents/Login"));
  const Home = lazy(() => import("./Screens/ListingScreen"));
  const NotFound = lazy(() => import("././Screens/ListingScreen"));
  return (
    <>
      {/* <Login /> */}
      {/* <ListingScreen /> */}
      <Router>
        <Suspense fallback={<div>Loading...</div>}>
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/home" element={<Home />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Router>
    </>
  );
}

export default App;
