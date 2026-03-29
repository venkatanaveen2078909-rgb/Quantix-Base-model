# 🌌 Quantix — Unified Quantum-Hybrid Optimization Platform (v3.0.0)

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue.svg)](https://vitejs.dev/)
[![Quantum](https://img.shields.io/badge/Solvers-Quantum--Hybrid-purple.svg)](https://www.dwavesys.com/)

**Quantix** is a production-ready, hybrid quantum-classical optimization suite that integrates state-of-the-art quantum algorithms with classical heuristics to solve complex business problems across three core domains: **Logistics**, **Supply Chain Management**, and **Investment Portfolios**.

---

## 🚀 Key Features

- **🌐 Unified Backend:** A consolidated FastAPI service handling multiple optimization domains via a single API.
- **⚡ Quantum-Classical Hybrid Solvers:**
    - **Logistics:** Optimized VRP (Vehicle Routing Problem) using D-Wave Tabu/SA and Simulated Annealing.
    - **Supply Chain:** Strategic network optimization using QAOA (Quantum Approximate Optimization Algorithm).
    - **Portfolio:** Risk-adjusted asset allocation using VQE (Variational Quantum Eigensolver).
- **📊 Real-time Dashboard:** A modern, responsive React/Shadcn UI for visualizing optimization results and business impact.
- **🧠 LLM Integration:** Automated analysis of optimization outputs via Groq/OpenAI compatible models.

---

## 🛠️ Technology Stack

### **Backend**
- **Core:** FastAPI, Python 3.14+
- **Database:** SQLite (Async) via SQLAlchemy/Aiosqlite
- **Solvers:** Qiskit (QAOA/VQE), D-Wave Ocean SDK (Hybrid Samplers), PuLP (Classical MILP)
- **Security:** JWT Authentication & CORS-ready

### **Frontend**
- **Core:** React, TypeScript, Vite
- **Styling:** TailwindCSS, Shadcn UI, Framer Motion
- **Visualization:** Recharts, Lucide Icons

---

## ⚙️ Installation & Setup

### **1. Backend Setup**
1. **Navigate to backend directory:**
   ```bash
   cd quantix-backend
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and add your API keys:
   ```env
   GROQ_API_KEY=your_key_here
   DWAVE_API_TOKEN=your_token_here
   ```
5. **Start the server:**
   ```bash
   python app.py
   ```
   *The backend will be live at `http://127.0.0.1:8000/docs` (Swagger UI).*

---

## 🛣️ API Endpoints (v1)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/logistics/optimize` | `POST` | Route & Fleet optimization |
| `/api/v1/supply-chain/optimize` | `POST` | Warehouse & Inventory allocation |
| `/api/v1/portfolio/optimize` | `POST` | Risk vs. Return asset targeting |
| `/health` | `GET` | System status check |

---

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any features or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
**Maintained by:** [venkatanaveen2078909](https://github.com/venkatanaveen2078909-rgb)
