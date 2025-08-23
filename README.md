<div align="center">

# 🧠 Mindful Focus 🧠

</div>

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Progress-blue?style=for-the-badge)
![Team](https://img.shields.io/badge/Team-NeuroGO-brightgreen?style=for-the-badge)
![Framework](https://img.shields.io/badge/Methodology-Kanban-purple?style=for-the-badge)
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

As a lean two-person team, our roles are fluid and collaborative, focused on getting the right work done.

| Role | Member | Focus Areas & Responsibilities |
| :--- | :--- | :--- |
| **Product & Tech Lead** | **Felix** | Guides the project vision, manages priorities, and leads the development of the data and AI pipeline. |
| **Lead Developer** | **Christie** | Leads the end-to-end application development, focusing on the user interface, backend integration, and overall user experience. |

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

Our plan is a living backlog of prioritized tasks. We focus on a continuous flow of development, pulling the next most important task as soon as we have the capacity to work on it.

The work is broadly grouped into two logical phases:

1.  **Core Technology:** Features related to building and validating the computer vision model and data pipeline.
2.  **Application & UX:** Features related to the user interface, gamification, and preparing a polished demo.

We will start with "Core Technology" priorities and fluidly transition to "Application & UX" tasks as the project matures.

---

## ⚠️ 5. Risk

A proactive approach to risk management is essential.

<details>
<summary><strong>🚨 View Risk Register</strong></summary>

| Risk ID | Risk Description | Response Plan |
| ------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| **R01** | Inaccurate AI readings due to variable lighting. | Implement a simple, one-time calibration step on app startup.              |
| **R02** | High CPU usage slows down the user's computer. | Optimize frame processing rate to once every 2 seconds instead of continuously. |
| **R03** | Unplanned work distracts from the main goal. | New ideas are added to the backlog and prioritized. We will complete current work before starting new tasks. |

</details>

---

## 🔄 6. Change

Our workflow is designed to embrace change. New ideas or requirement adjustments are added to the backlog, prioritized against existing tasks by the Product & Tech Lead, and pulled into the workflow as capacity allows.

---

## 📊 7. Progress

Progress is tracked visually to ensure transparency and maintain momentum.

* **Kanban Board:** A board in **Focalboard** is our single source of truth, with columns like `Backlog`, `To Do`, `In Progress`, and `Done`.
* **Focus on Finishing:** To ensure speed and prevent bottlenecks, we will **limit our 'In Progress' tasks to one per person**. A new task is only started once the current one is finished.
* **Process Visualization:** Key workflows are mapped in **Draw.io (diagrams.net)** to ensure clarity.
* **Daily Check-ins:** We will hold quick 15-minute syncs twice a day to discuss progress and clear any blockers.

---

## 🌊 Our Development Approach

Our workflow is built on three core principles inspired by the Kanban method, emphasizing flexibility, focus, and a continuous flow of value.

1.  **Visualize Everything**
    Our board in Focalboard makes our entire workflow visible. This transparency allows us to see progress in real-time, instantly spot bottlenecks, and make informed decisions about what to work on next.

2.  **Limit Work in Progress (WIP)**
    By strictly limiting the number of tasks we have "In Progress" at any one time, we improve our focus, increase the quality of our work, and dramatically speed up our delivery time. Our rule is simple: **Finish what you started before starting something new.**

3.  **Focus on Continuous Delivery**
    We don't work in rigid, time-boxed sprints. Our goal is a constant, smooth flow of completed tasks. We pull the highest-priority item from the backlog, finish it, and deliver it. This makes us highly adaptive and ensures we are always working on the most valuable feature at any given moment.