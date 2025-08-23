<div align="center">

# 🧠 Mindful Focus 🧠

</div>

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Progress-blue?style=for-the-badge)
![Team](https://img.shields.io/badge/Team-NeuroGO-brightgreen?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PRINCE2%20Agile-informational?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📄 Project Summary

> **Mindful Focus** is a desktop application designed to combat employee burnout and digital eye strain. Using a standard webcam and local AI processing, it passively monitors user well-being by analyzing metrics like **pupil dilation** (cognitive load) and **blink rate** (eye strain). It provides gentle, gamified nudges to encourage healthier work habits, transforming wellness from a chore into an engaging, positive experience.

---

## 🎯 1. Business Case

The objective is to develop a winning hackathon prototype that demonstrates a commercially viable solution for the growing corporate wellness market. Success means creating a functional, privacy-first MVP that can later be monetized through a freemium model.

### 💰 Monetization Strategy (Post-Hackathon)
While core features will remain free, a premium tier will be offered for teams and professionals. The payment and subscription infrastructure will be powered by **Unibee**.

* **Premium Features:**
    * Team Dashboards & Anonymized Analytics
    * Advanced Personal Analytics & Historical Trends
    * Integrations with Calendar Apps (Google, Outlook)
* **Payment Model:** A recurring monthly/annual subscription (SaaS).

---

## 🏢 2. Organization: Team NeuroGO

For a lean two-person team, members wear multiple hats, covering all essential project roles.

| Role | Assigned To | Key Responsibilities & Specialization |
| :--- | :--- | :--- |
| **Executive** & <br> **Project Manager** & <br> **Senior Supplier** | **Felix** | As the **Technopreneur & Project Lead**, Felix sets the project vision, ensures it's technically feasible, and manages the day-to-day schedule. He is the specialist for the **Data and AI Flow Engineering**. |
| **Senior User** & <br> **Lead Developer** | **Christie** | As the **Full Stack App Developer**, Christie builds the product and represents the end-user's perspective, ensuring the application is intuitive, functional, and meets user needs. |

#### A Note on Analyst Roles
In a two-person team, formal roles like **Business Analyst** and **Integration Analyst** are absorbed:
* **Business Analysis** is covered by Felix's technopreneurial role, defining the "what" and "why" of the project.
* **Integration Analysis** is handled by Christie as a full-stack developer, ensuring all parts of the application connect seamlessly.

---

## ✅ 3. Quality

Our definition of quality is a **stable, functional, and impressive demo**.

### Acceptance Criteria
- [ ] Application successfully accesses the webcam with one-click user permission.
- [ ] Real-time dashboard displays a "Focus Score" and "Eye Strain" meter.
- [ ] AI model correctly identifies blink rate and relative pupil size.
- [ ] Gamification engine awards points for following break suggestions.
- [ ] All data processing is confirmed to be 100% local for privacy.
- [ ] The app runs for the full 5-minute demo without any crashes.

---

## 🗺️ 4. Plans

The project is divided into two time-boxed stages to ensure key milestones are met.

<details>
<summary><strong>📋 Stage 1: Proof of Concept (Hours 1-24)</strong></summary>

The primary goal of this stage is to validate the core technology. The deliverable is an internal build where the AI model successfully feeds data to the application backend.
* **Hours 1-2:** Setup & Initiation (Git repo, **Focalboard** project).
* **Hours 3-12:** Core Model Development (Python w/ OpenCV, MediaPipe).
* **Hours 13-24:** Backend API & Data Handling (e.g., FastAPI).
</details>

<details>
<summary><strong>🚀 Stage 2: Productization & Demo Prep (Hours 25-48)</strong></summary>

With a working backend, this stage focuses on building the user-facing application and ensuring it's ready for the final presentation.
* **Hours 25-36:** UI/UX Development (Electron, React/Svelte).
* **Hours 37-44:** Integration, Testing, and Bug Fixing.
* **Hours 45-48:** Finalize Presentation & Practice Demo.
</details>

---

## ⚠️ 5. Risk

A proactive approach to risk management is essential.

<details>
<summary><strong>🚨 View Risk Register</strong></summary>

| Risk ID | Risk Description | Response Plan |
| ------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| **R01** | Inaccurate AI readings due to variable lighting. | Implement a simple, one-time calibration step on app startup. |
| **R02** | High CPU usage slows down the user's computer. | Optimize frame processing rate to once every 2 seconds instead of continuously. |
| **R03** | Scope creep from new ideas mid-hackathon. | The Project Manager will enforce the initial feature list. New ideas are logged in a backlog. |

</details>

---

## 🔄 6. Change

Change will be handled via a lightweight control process: any proposed change must be presented to the **Project Manager**, who will make a rapid go/no-go decision after assessing its impact on the deadline.

---

## 📊 7. Progress

Progress will be tracked visually using open-source tools to maintain momentum.

* **Task Management:** A Kanban board in **Focalboard** will be our single source of truth for task status (`To Do`, `In Progress`, `Done`).
* **Process Visualization:** Key workflows will be mapped in **Draw.io (diagrams.net)** to ensure clarity for all team members.
* **Check-ins:** The team will hold 15-minute stand-ups every 6 hours to sync up, report progress, and identify blockers.

---

## 💡 PRINCE2 & Iterative Development

**Traditionally, PRINCE2 itself is not iterative.** It focuses on high-level direction and control. However, it is designed to be flexible and is commonly integrated with agile methods in an approach known as **PRINCE2 Agile®**.

Think of it this way:
* **PRINCE2** is the strategic **mission command**, answering "What should we do?" and "Why?"
* **Agile** methods are the tactical **squad on the ground**, answering "How will we build it in short, flexible cycles?"

### How It Applies to This Hackathon
Our project plan uses this hybrid model:

1.  **PRINCE2 Management Stages:** We've broken the 48-hour hackathon into two high-level stages (`Stage 1: Proof of Concept` and `Stage 2: Productization`). The decision to move between stages is a formal control point.

2.  **Iterative Work Within Stages:** Inside each stage, your team will work in short **iterations** (or "sprints"). For example, within Stage 2, you could have:
    * **Iteration 1 (4 hours):** Build the basic dashboard UI.
    * **Iteration 2 (6 hours):** Integrate the AI data stream.
    * **Iteration 3 (4 hours):** Implement the gamification logic.

This approach provides both the high-level control of PRINCE2 and the on-the-ground flexibility of iterative development, which is perfect for a hackathon.