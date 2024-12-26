import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import './SignUp.css'; // Importing the CSS file for styling

const SignUpPage = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/api/auth/signup", { name, email, password, role });
      localStorage.setItem("token", response.data.token);
      navigate("/home"); // Redirect to home page on success
    } catch (error) {
      alert("Error signing up");
    }
  };

  return (
    <div className="signup-wrapper">
      <div className="signup-container">
        <h2 className="signup-header">Sign Up</h2>
        <form onSubmit={handleSubmit} className="signup-form">
          <input
            type="text"
            className="signup-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Name"
            required
          />
          <input
            type="email"
            className="signup-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            required
          />
          <input
            type="password"
            className="signup-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            required
          />
          <select
            className="signup-select"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            required
          >
            <option value="">Choose a role</option>
            <option value="Teacher">Teacher</option>
            <option value="Principal">Principal</option>
            <option value="Admin">Admin</option>
          </select>
          <button type="submit" className="signup-button">
            Sign Up
          </button>
        </form>
        <p className="signup-footer">
          Already have an account? <Link to="/" className="signup-link">Login</Link>
        </p>
      </div>
    </div>
  );
};

export default SignUpPage;
